#!/usr/bin/env python3

"""
Create EDFbrowser-compatible TXT annotation file.

Output format (STRICT):
<onset> <duration> <label>

Example:
0 10 AF
10 10 Normal
"""

import pandas as pd
import os
import argparse


def extract_times(fname):
    """
    Extract start and end timestamps (ns) from filename
    Format: patient_start_end.mat
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("_")

    try:
        start_ns = int(parts[-2])
        end_ns = int(parts[-1])
    except Exception:
        raise ValueError(f"Invalid filename format: {fname}")

    return start_ns, end_ns


# def create_annotations_txt(pred_csv, out_txt):
#     print("\n[STEP 1] Loading prediction CSV...")
#     df = pd.read_csv(pred_csv)
#
#     if "file_name" not in df.columns or "pred_label" not in df.columns:
#         raise ValueError("CSV must contain 'file_name' and 'pred_label'")
#
#     print(f"Total rows: {len(df)}")
#
#     print("\n[STEP 2] Extracting timestamps...")
#     df[["start_ns", "end_ns"]] = df["file_name"].apply(
#         lambda x: pd.Series(extract_times(x))
#     )
#
#     print("\n[STEP 3] Convert to seconds...")
#     df["start_sec"] = df["start_ns"] / 1e9
#     df["end_sec"] = df["end_ns"] / 1e9
#
#     print("\n[STEP 4] Normalize time...")
#     t0 = df["start_sec"].min()
#     df["onset"] = df["start_sec"] - t0
#
#     print("\n[STEP 5] Compute duration...")
#     df["duration"] = df["end_sec"] - df["start_sec"]
#
#     print("\n[STEP 6] Map labels...")
#     label_map = {
#         0: "NotAF",
#         1: "AF"
#     }
#
#     df["label"] = df["pred_label"].map(label_map).fillna(
#         df["pred_label"].astype(str)
#     )
#
#     print("\n[STEP 7] Sort by time...")
#     df = df.sort_values("onset").reset_index(drop=True)
#
#     print("\n[STEP 8] Writing TXT file...")
#
#     with open(out_txt, "w") as f:
#         for _, row in df.iterrows():
#             onset = round(row["onset"], 3)
#             duration = round(row["duration"], 3)
#             label = row["label"]
#
#             # 🔥 IMPORTANT: add "+"
#             f.write(f"+{onset} {duration} {label}\n")
#
#     print("\n================ DONE =================")
#     print(f"Saved TXT annotations: {out_txt}")
#     print(f"Total annotations: {len(df)}")

def create_annotations_txt(pred_csv, out_txt):
    print("\n[STEP 1] Loading prediction CSV...")
    df = pd.read_csv(pred_csv)

    if "file_name" not in df.columns or "pred_label" not in df.columns:
        raise ValueError("CSV must contain 'file_name' and 'pred_label'")

    print(f"Total rows: {len(df)}")

    print("\n[STEP 2] Extracting timestamps...")
    df[["start_ns", "end_ns"]] = df["file_name"].apply(
        lambda x: pd.Series(extract_times(x))
    )

    print("\n[STEP 3] Convert to seconds...")
    df["start_sec"] = df["start_ns"] / 1e9
    df["end_sec"] = df["end_ns"] / 1e9

    print("\n[STEP 4] Normalize time...")
    t0 = df["start_sec"].min()
    df["onset"] = df["start_sec"] - t0

    print("\n[STEP 5] Compute duration...")
    df["duration"] = df["end_sec"] - df["start_sec"]

    print("\n[STEP 6] Map labels...")
    label_map = {
        0: "NotAF",
        1: "AF"
    }

    df["label"] = df["pred_label"].map(label_map).fillna(
        df["pred_label"].astype(str)
    )

    print("\n[STEP 7] Sort by time...")
    df = df.sort_values("onset").reset_index(drop=True)

    print("\n[STEP 8] Writing ASCII annotation file...")

    # column-based ASCII file for EDFbrowser
    ann_df = df[["onset", "duration", "label"]].copy()
    ann_df["onset"] = ann_df["onset"].round(3)
    ann_df["duration"] = ann_df["duration"].round(3)

    # save as comma-separated text
    ann_df.to_csv(out_txt, index=False, sep=",", lineterminator="\n")

    print("\n================ DONE =================")
    print(f"Saved TXT annotations: {out_txt}")
    print(f"Total annotations: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create EDFbrowser TXT annotations"
    )

    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--out_txt", type=str, required=True)

    args = parser.parse_args()

    create_annotations_txt(args.pred_csv, args.out_txt)


if __name__ == "__main__":
    main()
