import os
import glob
import numpy as np
import pandas as pd
import wfdb
from scipy.io import savemat

# -----------------------------
# Helpers: parse rhythm labels
# -----------------------------
def extract_rhythm_intervals(ann, n_samples):
    """
    Build (start, end, label) intervals from MIT-BIH AF rhythm annotations.
    label in {"AF", "NotAF"}.

    MIT-BIH AF uses rhythm annotations in ann.aux_note like:
      "(AFIB", "(AFL", "(N", etc.
    We'll treat AFIB/AFL as AF, everything else as NotAF.
    """
    samples = np.array(ann.sample, dtype=int)
    aux = [a.strip() if isinstance(a, str) else "" for a in ann.aux_note]

    # Some records may have no rhythm aux_note; handle defensively
    # We'll build change-points where aux_note indicates a rhythm start.
    change_points = []
    labels = []

    for s, note in zip(samples, aux):
        if not note:
            continue
        # Rhythm annotations often start with '('
        # Example: "(AFIB", "(N"
        if note.startswith("("):
            rhythm = note[1:].strip().upper()
            if rhythm in ["AFIB", "AF", "AFL"]:
                lab = "AF"
            else:
                lab = "NotAF"
            change_points.append(s)
            labels.append(lab)

    # If we found no rhythm markers, default whole record to NotAF
    if len(change_points) == 0:
        return [(0, n_samples, "NotAF")]

    # Ensure sorted by time
    order = np.argsort(change_points)
    change_points = [change_points[i] for i in order]
    labels = [labels[i] for i in order]

    # Build intervals
    intervals = []
    for i in range(len(change_points)):
        start = change_points[i]
        end = change_points[i + 1] if i + 1 < len(change_points) else n_samples
        if end > start:
            intervals.append((start, end, labels[i]))

    # Merge consecutive intervals with same label (just in case)
    merged = []
    for st, en, lab in intervals:
        if not merged:
            merged.append([st, en, lab])
        else:
            pst, pen, plab = merged[-1]
            if plab == lab and st <= pen:
                merged[-1][1] = max(pen, en)
            else:
                merged.append([st, en, lab])

    return [(a, b, c) for a, b, c in merged]


def segment_interval(start, end, fs, label, win_sec=10.0):
    """
    Generate (seg_start, seg_end) windows fully inside [start,end).
    NotAF: stride = 10s
    AF: stride = 8s (2s overlap)
    AF interval shorter than 10s -> single window with actual length (short).
    NotAF interval shorter than 10s -> produce none (can't make a full 10s patch).
    """
    win = int(round(win_sec * fs))
    length = end - start

    if label == "AF":
        if length < win:
            return [(start, end)]  # short AF episode kept as one segment
        # stride = int(round((win_sec - 2.0) * fs))  # 8 seconds
        stride = win  # 10 seconds

    else:
        # NotAF
        if length < win:
            return []
        stride = win  # 10 seconds

    segs = []
    cur = start
    while cur + win <= end:
        segs.append((cur, cur + win))
        cur += stride

    return segs


# -----------------------------
# Main
# -----------------------------
def process_record(record_path_no_ext, out_dir, rows, win_sec=10.0):
    # Read waveform
    sig, fields = wfdb.rdsamp(record_path_no_ext)
    fs = float(fields["fs"])
    n_samples = sig.shape[0]

    # Read annotations (.atr)
    ann = wfdb.rdann(record_path_no_ext, "atr")

    intervals = extract_rhythm_intervals(ann, n_samples)

    record_name = os.path.basename(record_path_no_ext)
    rec_out = os.path.join(out_dir, record_name)
    os.makedirs(rec_out, exist_ok=True)

    seg_idx = 0
    for st, en, lab in intervals:
        seg_ranges = segment_interval(st, en, fs, lab, win_sec=win_sec)

        for a, b in seg_ranges:
            segment = sig[a:b, :]  # (samples, channels)

            seg_name = f"{record_name}_{seg_idx:05d}"
            mat_path = os.path.join(rec_out, f"{seg_name}.mat")

            savemat(
                mat_path,
                {
                    "ecg": segment.astype(np.float32),
                    "fs": np.array([fs], dtype=np.float32),
                    "label": np.array([lab], dtype=object),
                    "record": np.array([record_name], dtype=object),
                    "start_sample": np.array([a], dtype=np.int64),
                    "end_sample": np.array([b], dtype=np.int64),
                },
                do_compression=True,
            )

            rows.append(
                {
                    "segment_name": seg_name,
                    "record_name": record_name,
                    "label": lab,
                    "fs": fs,
                    "start_sample": a,
                    "end_sample": b,
                    "num_samples": int(b - a),
                    "mat_path": os.path.relpath(mat_path, out_dir),
                }
            )
            seg_idx += 1


def main(in_dir, out_dir, win_sec=10.0):
    os.makedirs(out_dir, exist_ok=True)

    # Each record has .hea; use that to enumerate records
    hea_files = sorted(glob.glob(os.path.join(in_dir, "*.hea")))
    if not hea_files:
        raise RuntimeError(f"No .hea files found in: {in_dir}")

    rows = []
    for hea in hea_files:
        rec = os.path.splitext(os.path.basename(hea))[0]
        record_path_no_ext = os.path.join(in_dir, rec)

        # Ensure corresponding .dat and .atr exist
        if not os.path.exists(record_path_no_ext + ".dat"):
            print(f"[SKIP] Missing .dat for {rec}")
            continue
        if not os.path.exists(record_path_no_ext + ".atr"):
            print(f"[SKIP] Missing .atr for {rec}")
            continue

        print(f"[OK] Processing: {rec}")
        process_record(record_path_no_ext, out_dir, rows, win_sec=win_sec)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "labels.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Total segments: {len(df)}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help="Folder containing .hea/.dat/.atr")
    p.add_argument("--out_dir", required=True, help="Output folder")
    p.add_argument("--win_sec", type=float, default=10.0, help="Window length in seconds (default: 10)")
    args = p.parse_args()

    main(args.in_dir, args.out_dir, win_sec=args.win_sec)
