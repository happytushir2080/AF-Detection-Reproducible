import os
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, lfilter, iirnotch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utils
# =========================================================
def _filtfilt_or_lfilter(b, a, x):
    """
    Try filtfilt first for zero-phase filtering.
    If signal is too short, fall back to lfilter.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    try:
        return filtfilt(b, a, x).astype(np.float32)
    except ValueError:
        # too short for filtfilt padlen
        return lfilter(b, a, x).astype(np.float32)


def bandpass_filter(x, fs, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq

    # safety clamp
    low_n = max(low_n, 1e-6)
    high_n = min(high_n, 0.999)

    if not (0 < low_n < high_n < 1):
        raise ValueError(
            f"Invalid bandpass range after normalization: low={low_n}, high={high_n}, fs={fs}"
        )

    b, a = butter(order, [low_n, high_n], btype="band")
    return _filtfilt_or_lfilter(b, a, x)


def notch_filter(x, fs, freq=50.0, Q=30.0):
    w0 = freq / (0.5 * fs)
    if not (0 < w0 < 1):
        warnings.warn(
            f"Skipping notch filter because normalized freq is invalid: w0={w0:.4f}, fs={fs}, freq={freq}"
        )
        return np.asarray(x, dtype=np.float32)
    b, a = iirnotch(w0, Q)
    return _filtfilt_or_lfilter(b, a, x)


def ecg_log_spectrogram(x, fs, nperseg=64, noverlap=32, alpha=1e-6):
    """
    Returns log power spectrogram S: (F, T)
    Using torch.stft for consistency + speed.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    xt = torch.tensor(x, dtype=torch.float32)

    if len(x) < nperseg:
        pad_len = nperseg - len(x)
        xt = F.pad(xt, (0, pad_len))

    window = torch.hann_window(nperseg)

    Z = torch.stft(
        xt,
        n_fft=nperseg,
        hop_length=nperseg - noverlap,
        win_length=nperseg,
        window=window,
        center=False,
        return_complex=True
    )
    P = (Z.real ** 2 + Z.imag ** 2)
    S = torch.log(P + alpha)
    return S.numpy().astype(np.float32)


def load_window_mat(mat_path):
    """
    Loads waveform from your saved MAT files.

    Expected:
      signal: (samples, 1) or (samples,) or (samples, channels)

    Returns:
      x: 1D numpy array
      fs: float
      meta: dict
    """
    d = loadmat(mat_path)

    if "signal" not in d:
        raise KeyError(f"'signal' not found in {mat_path}")

    x = d["signal"]

    # handle shapes
    # (N,1) -> (N,)
    # (1,N) -> (N,)
    # (N,C) -> use first channel
    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x[:, 0]
        elif x.shape[0] == 1:
            x = x[0, :]
        else:
            # if multi-channel accidentally saved, use first channel
            x = x[:, 0]
    else:
        x = x.reshape(-1)

    x = np.asarray(x, dtype=np.float32)

    if "fs" not in d:
        raise KeyError(f"'fs' not found in {mat_path}")
    fs = float(np.array(d["fs"]).reshape(-1)[0])

    meta = {}
    for k in ["patient_name", "lead_name", "start_time_ns", "end_time_ns", "start_sec", "end_sec"]:
        if k in d:
            try:
                meta[k] = np.array(d[k]).reshape(-1)[0]
            except Exception:
                meta[k] = d[k]

    return x, fs, meta


# =========================================================
# Dataset
# =========================================================
class MatInferenceDataset(Dataset):
    def __init__(
        self,
        mat_files,
        use_notch=False,
        spec_nperseg=64,
        spec_noverlap=32,
        spec_alpha=1e-6,
        verbose=False,
    ):
        self.mat_files = list(mat_files)
        self.use_notch = use_notch
        self.spec_nperseg = spec_nperseg
        self.spec_noverlap = spec_noverlap
        self.spec_alpha = spec_alpha
        self.verbose = verbose

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        mat_path = self.mat_files[idx]
        file_name = os.path.basename(mat_path)

        x, fs, meta = load_window_mat(mat_path)

        if self.verbose and idx < 5:
            print(f"[DATASET] Loading {file_name} | len={len(x)} | fs={fs}")

        # filtering
        x = bandpass_filter(x, fs=fs, low=0.5, high=40.0, order=4)
        if self.use_notch:
            x = notch_filter(x, fs=fs, freq=50.0, Q=30.0)

        # normalize waveform
        x = (x - x.mean()) / (x.std() + 1e-6)

        # spectrogram
        S = ecg_log_spectrogram(
            x, fs=fs,
            nperseg=self.spec_nperseg,
            noverlap=self.spec_noverlap,
            alpha=self.spec_alpha
        )

        # normalize spectrogram
        S = (S - S.mean()) / (S.std() + 1e-6)

        return {
            "file_name": file_name,
            "mat_path": mat_path,
            "spec": torch.tensor(S, dtype=torch.float32),
            "fs": fs,
            "meta": meta,
        }


def collate_pad_infer(batch):
    file_names = [b["file_name"] for b in batch]
    mat_paths = [b["mat_path"] for b in batch]
    fss = [b["fs"] for b in batch]
    metas = [b["meta"] for b in batch]
    specs = [b["spec"] for b in batch]

    Fdim = specs[0].shape[0]
    T_max = max(s.shape[1] for s in specs)

    X = torch.zeros(len(batch), 1, Fdim, T_max, dtype=torch.float32)
    mask = torch.zeros(len(batch), T_max, dtype=torch.bool)

    for i, S in enumerate(specs):
        T = S.shape[1]
        X[i, 0, :, :T] = S
        mask[i, :T] = True

    return {
        "file_names": file_names,
        "mat_paths": mat_paths,
        "X": X,
        "mask": mask,
        "fs": fss,
        "meta": metas,
    }


# =========================================================
# Model
# =========================================================
def masked_mean_pooling(x, mask):
    m = mask.unsqueeze(-1).float()
    x = x * m
    return x.sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def masked_topk_pooling(x, mask, k=5):
    neg_inf = torch.tensor(-1e9, device=x.device, dtype=x.dtype)
    x = torch.where(mask.unsqueeze(-1), x, neg_inf)
    k_eff = min(k, x.shape[1])
    topk_vals, _ = torch.topk(x, k=k_eff, dim=1)
    return topk_vals.mean(dim=1)


class SpecCRNNMaskedPoolNet(nn.Module):
    def __init__(self, dropout=0.3, lstm_hidden=64, feat_dim=32, topk=5):
        super().__init__()
        self.topk = topk

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=64 * 8,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.feat_drop = nn.Dropout(dropout)
        self.feat_fc = nn.Linear(2 * lstm_hidden * 2, feat_dim)
        self.out_drop = nn.Dropout(dropout)
        self.out_fc = nn.Linear(feat_dim, 1)

    def forward(self, X, mask):
        z = self.conv(X)
        B, C, Fp, T = z.shape

        z = z.permute(0, 3, 1, 2).contiguous().view(B, T, C * Fp)

        out, _ = self.lstm(z)

        mean_pool = masked_mean_pooling(out, mask)
        topk_pool = masked_topk_pooling(out, mask, k=self.topk)
        pooled = torch.cat([mean_pool, topk_pool], dim=1)

        feat = F.relu(self.feat_fc(self.feat_drop(pooled)))
        logit = self.out_fc(self.out_drop(feat)).squeeze(-1)
        return logit


# =========================================================
# Inference
# =========================================================
def find_mat_files(input_dir):
    return sorted(glob.glob(os.path.join(input_dir, "*.mat")))


def load_model(checkpoint_path, device, dropout=0.3, lstm_hidden=64, feat_dim=32, topk=5):
    print(f"[MODEL] Building model...")
    model = SpecCRNNMaskedPoolNet(
        dropout=dropout,
        lstm_hidden=lstm_hidden,
        feat_dim=feat_dim,
        topk=topk
    ).to(device)

    print(f"[MODEL] Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("[MODEL] Checkpoint loaded successfully.")
    return model


@torch.no_grad()
def run_inference(model, loader, device, threshold=0.5):
    rows = []
    total = 0

    for batch_idx, batch in enumerate(loader):
        X = batch["X"].to(device)
        mask = batch["mask"].to(device)

        logits = model(X, mask)
        probs = torch.sigmoid(logits).cpu().numpy()

        file_names = batch["file_names"]
        mat_paths = batch["mat_paths"]
        fss = batch["fs"]
        metas = batch["meta"]

        for i in range(len(file_names)):
            prob = float(probs[i])
            pred = 1 if prob >= threshold else 0
            pred_label = "AF" if pred == 1 else "NotAF"

            row = {
                "file_name": file_names[i],
                "mat_path": mat_paths[i],
                "fs": fss[i],
                "prob_af": prob,
                "pred_label": pred_label,
                "threshold_used": threshold,
            }

            meta = metas[i]
            for k, v in meta.items():
                row[k] = v

            rows.append(row)
            total += 1

        if batch_idx % 10 == 0:
            print(f"[INFER] Processed batch {batch_idx} | total files so far: {total}")

    return pd.DataFrame(rows)


# =========================================================
# Main
# =========================================================
def main(args):
    print("\n================ INFERENCE START ================")
    print(f"Input dir      : {args.input_dir}")
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Output csv     : {args.output_csv}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Threshold      : {args.threshold}")
    print(f"Use notch      : {args.use_notch}")
    print(f"Num workers    : {args.num_workers}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[DEVICE] Using CPU")

    mat_files = find_mat_files(args.input_dir)
    print(f"[DATA] Found {len(mat_files)} .mat files")

    if len(mat_files) == 0:
        raise RuntimeError(f"No .mat files found in {args.input_dir}")

    print("[DATA] First 5 files:")
    for p in mat_files[:5]:
        print("   ", os.path.basename(p))

    dataset = MatInferenceDataset(
        mat_files=mat_files,
        use_notch=args.use_notch,
        spec_nperseg=args.spec_nperseg,
        spec_noverlap=args.spec_noverlap,
        spec_alpha=args.spec_alpha,
        verbose=args.verbose,
    )

    sample = dataset[0]

    print("\nKeys:", sample.keys())

    print("\nFile name:", sample["file_name"])
    print("Mat path:", sample["mat_path"])

    print("\nSpectrogram shape:", sample["spec"].shape)
    print("Spectrogram dtype:", sample["spec"].dtype)

    print("\nSampling frequency (fs):", sample["fs"])

    print("\nMeta:", sample["meta"])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_pad_infer,
    )

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        dropout=args.dropout,
        lstm_hidden=args.lstm_hidden,
        feat_dim=args.feat_dim,
        topk=args.topk,
    )

    out_df = run_inference(
        model=model,
        loader=loader,
        device=device,
        threshold=args.threshold,
    )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print("\n================ INFERENCE SUMMARY ================")
    print(f"Total files inferred: {len(out_df)}")
    print("\nPrediction counts:")
    print(out_df["pred_label"].value_counts(dropna=False))

    print("\nProbability stats:")
    print(out_df["prob_af"].describe())

    print(f"\nSaved results to: {args.output_csv}")
    print("================ INFERENCE DONE ================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder containing .mat files")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_csv", type=str, default="./inference_results.csv",
                        help="Where to save predictions CSV")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--use_notch", action="store_true")

    parser.add_argument("--spec_nperseg", type=int, default=64)
    parser.add_argument("--spec_noverlap", type=int, default=32)
    parser.add_argument("--spec_alpha", type=float, default=1e-6)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--feat_dim", type=int, default=32)
    parser.add_argument("--topk", type=int, default=5)

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)



# python inference.py \
#   --input_dir ./AIIMS_data/Danish Holter_1\
#   --checkpoint ./code-7_patient_split/best_spec_crnn.pt \
#   --output_csv dansih_test.csv \
#   --batch_size 32 \
#   --threshold 0.9891 \
#   --use_notch

