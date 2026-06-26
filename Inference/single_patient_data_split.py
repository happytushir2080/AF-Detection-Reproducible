# import os
# import numpy as np
# import pandas as pd
# from scipy.io import savemat

# # -------------------------
# # INPUT / OUTPUT
# # -------------------------
# CSV_PATH = "/mnt/project2/DST-SERB/AIIMS-data/AIIMS_recordings/Ramesh/1/Holter/24_06_2025.csv"   # change this
# OUT_DIR = "./ramesh_patient_lead2_10s_mat"
# os.makedirs(OUT_DIR, exist_ok=True)

# WIN_SEC = 10.0
# TIME_COL = "time"
# LEAD_COL = "Lead_2"   # Lead II
# PATIENT_NAME = os.path.splitext(os.path.basename(CSV_PATH))[0]

# print("\n================ START PROCESSING =================")
# print(f"Patient: {PATIENT_NAME}")
# print(f"CSV Path: {CSV_PATH}")
# print(f"Output Dir: {OUT_DIR}")

# # -------------------------
# # Load CSV
# # -------------------------
# print("\n[STEP 1] Loading CSV...")
# df = pd.read_csv(CSV_PATH)

# print(f"Initial rows: {len(df)}")

# df = df[[TIME_COL, LEAD_COL]].copy()
# df = df.sort_values(TIME_COL).reset_index(drop=True)
# df = df.dropna(subset=[TIME_COL, LEAD_COL]).reset_index(drop=True)

# print(f"Rows after cleaning: {len(df)}")
# print(df.head())

# # -------------------------
# # Compute sampling rate
# # -------------------------
# print("\n[STEP 2] Computing sampling rate...")

# time_ns = df[TIME_COL].values.astype(np.int64)
# time_sec = time_ns / 1e9

# dt = np.diff(time_sec)
# fs = 1.0 / np.median(dt)

# print(f"Estimated fs = {fs:.4f} Hz")
# print(f"Median dt = {np.median(dt):.6f} sec")

# # -------------------------
# # Extract Lead II
# # -------------------------
# print("\n[STEP 3] Extracting Lead II...")

# signal = df[LEAD_COL].values.astype(np.float32).reshape(-1, 1)

# n_samples = signal.shape[0]
# win_len = int(round(WIN_SEC * fs))

# print(f"Signal shape: {signal.shape}")
# print(f"Samples per 10s window: {win_len}")

# total_possible_windows = n_samples // win_len
# print(f"Total possible full windows: {total_possible_windows}")

# # -------------------------
# # Save 10s windows
# # -------------------------
# print("\n[STEP 4] Creating and saving windows...")

# rows = []
# saved_count = 0
# skipped_zero_count = 0
# skipped_short_count = 0

# for i, start in enumerate(range(0, n_samples, win_len)):
#     end = start + win_len

#     # progress print every 100 windows
#     if i % 100 == 0:
#         print(f"Processing window {i}...")

#     # skip last incomplete window
#     if end > n_samples:
#         skipped_short_count += 1
#         print(f"[INFO] Skipping last incomplete window at index {i}")
#         break

#     segment = signal[start:end, :]

#     # skip all-zero segments
#     if np.all(segment == 0):
#         skipped_zero_count += 1
#         continue

#     start_time_ns = int(time_ns[start])
#     end_time_ns = int(time_ns[end - 1])

#     mat_name = f"{PATIENT_NAME}_{start_time_ns}_{end_time_ns}.mat"
#     mat_path = os.path.join(OUT_DIR, mat_name)

#     savemat(
#         mat_path,
#         {
#             "signal": segment,
#             "fs": np.array([fs], dtype=np.float32),
#             "lead_name": np.array([LEAD_COL], dtype=object),
#             "patient_name": np.array([PATIENT_NAME], dtype=object),
#             "start_sample": np.array([start], dtype=np.int64),
#             "end_sample": np.array([end], dtype=np.int64),
#             "start_time_ns": np.array([start_time_ns], dtype=np.int64),
#             "end_time_ns": np.array([end_time_ns], dtype=np.int64),
#             "start_sec": np.array([time_sec[start]], dtype=np.float32),
#             "end_sec": np.array([time_sec[end - 1]], dtype=np.float32),
#         },
#         do_compression=True,
#     )

#     rows.append({
#         "patient_name": PATIENT_NAME,
#         "file_name": mat_name,
#         "start_sample": start,
#         "end_sample": end,
#         "start_time_ns": start_time_ns,
#         "end_time_ns": end_time_ns,
#         "num_samples": end - start,
#         "fs": fs,
#         "duration_sec": (end - start) / fs,
#         "lead_used": LEAD_COL,
#         "mat_path": mat_path,
#     })

#     saved_count += 1

# # -------------------------
# # Save manifest
# # -------------------------
# print("\n[STEP 5] Saving manifest CSV...")

# manifest_df = pd.DataFrame(rows)
# manifest_csv = os.path.join(OUT_DIR, "windows_manifest.csv")
# manifest_df.to_csv(manifest_csv, index=False)

# # -------------------------
# # Final Summary
# # -------------------------
# print("\n================ SUMMARY =================")
# print(f"Total samples: {n_samples}")
# print(f"Window size (samples): {win_len}")
# print(f"Total possible windows: {total_possible_windows}")

# print(f"\nSaved windows: {saved_count}")
# print(f"Skipped all-zero windows: {skipped_zero_count}")
# print(f"Skipped incomplete windows: {skipped_short_count}")

# if saved_count > 0:
#     print("\nSample output:")
#     print(manifest_df.head())

# print("\nOutput directory:", OUT_DIR)
# print("Manifest file:", manifest_csv)

# print("\n================ DONE =================")

import os
import argparse
import numpy as np
import pandas as pd
from scipy.io import savemat
import sys


def setup_logger(out_dir):
    log_path = os.path.join(out_dir, "processing.log")

    class Logger:
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_path, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger()
    sys.stderr = sys.stdout

    return log_path


def main(args):
    CSV_PATH = args.csv_path
    OUT_DIR = args.out_dir

    os.makedirs(OUT_DIR, exist_ok=True)

    # 🔥 SETUP LOGGING
    log_path = setup_logger(OUT_DIR)

    WIN_SEC = 10.0
    TIME_COL = "time"
    LEAD_COL = "Lead_2"

    PATIENT_NAME = os.path.splitext(os.path.basename(CSV_PATH))[0]

    print("\n================ START PROCESSING =================")
    print(f"Patient: {PATIENT_NAME}")
    print(f"CSV Path: {CSV_PATH}")
    print(f"Output Dir: {OUT_DIR}")
    print(f"Log File: {log_path}")

    try:
        # -------------------------
        # Load CSV
        # -------------------------
        print("\n[STEP 1] Loading CSV...")
        df = pd.read_csv(CSV_PATH)

        print(f"Initial rows: {len(df)}")

        df = df[[TIME_COL, LEAD_COL]].copy()
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        df = df.dropna(subset=[TIME_COL, LEAD_COL]).reset_index(drop=True)

        print(f"Rows after cleaning: {len(df)}")

        # -------------------------
        # Compute sampling rate
        # -------------------------
        print("\n[STEP 2] Computing sampling rate...")

        time_ns = df[TIME_COL].values.astype(np.int64)
        time_sec = time_ns / 1e9

        dt = np.diff(time_sec)
        fs = 1.0 / np.median(dt)

        print(f"Estimated fs = {fs:.4f} Hz")

        # -------------------------
        # Extract signal
        # -------------------------
        print("\n[STEP 3] Extracting Lead II...")

        signal = df[LEAD_COL].values.astype(np.float32).reshape(-1, 1)

        n_samples = signal.shape[0]
        win_len = int(round(WIN_SEC * fs))

        print(f"Signal shape: {signal.shape}")
        print(f"Samples per 10s window: {win_len}")

        # -------------------------
        # Save windows
        # -------------------------
        print("\n[STEP 4] Creating and saving windows...")

        rows = []
        saved_count = 0
        skipped_zero_count = 0
        skipped_short_count = 0

        for i, start in enumerate(range(0, n_samples, win_len)):
            end = start + win_len

            if i % 100 == 0:
                print(f"Processing window {i}...")

            if end > n_samples:
                skipped_short_count += 1
                print(f"[INFO] Skipping incomplete window at index {i}")
                break

            segment = signal[start:end, :]

            if np.all(segment == 0):
                skipped_zero_count += 1
                continue

            start_time_ns = int(time_ns[start])
            end_time_ns = int(time_ns[end - 1])

            mat_name = f"{PATIENT_NAME}_{start_time_ns}_{end_time_ns}.mat"
            mat_path = os.path.join(OUT_DIR, mat_name)

            savemat(
                mat_path,
                {
                    "signal": segment,
                    "fs": np.array([fs], dtype=np.float32),
                    "patient_name": np.array([PATIENT_NAME], dtype=object),
                },
                do_compression=True,
            )

            rows.append({
                "patient_name": PATIENT_NAME,
                "file_name": mat_name,
                "mat_path": mat_path,
                "fs": fs
            })

            saved_count += 1

        # -------------------------
        # Save manifest
        # -------------------------
        print("\n[STEP 5] Saving manifest CSV...")

        manifest_df = pd.DataFrame(rows)
        manifest_csv = os.path.join(OUT_DIR, "windows_manifest.csv")
        manifest_df.to_csv(manifest_csv, index=False)

        # -------------------------
        # Summary
        # -------------------------
        print("\n================ SUMMARY =================")
        print(f"Total samples: {n_samples}")
        print(f"Saved windows: {saved_count}")
        print(f"Skipped zero: {skipped_zero_count}")
        print(f"Skipped short: {skipped_short_count}")

        print("\nOutput directory:", OUT_DIR)
        print("Manifest file:", manifest_csv)

    except Exception as e:
        print("\n[ERROR] Processing failed!")
        print(str(e))

    print("\n================ DONE =================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()
    main(args)