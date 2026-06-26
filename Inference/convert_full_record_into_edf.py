#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pyedflib
import argparse
import os


def estimate_fs(time_ns):
    time_sec = time_ns / 1e9
    dt = np.diff(time_sec)
    fs = 1.0 / np.median(dt)
    return fs


def convert_csv_to_edf(csv_path, out_edf):
    print("\n[STEP 1] Loading CSV...")
    df = pd.read_csv(csv_path)

    TIME_COL = "time"
    SIGNAL_COL = "Lead_2"

    df = df[[TIME_COL, SIGNAL_COL]].dropna().reset_index(drop=True)

    print(f"Total samples: {len(df)}")

    print("\n[STEP 2] Estimating sampling rate...")
    time_ns = df[TIME_COL].values.astype(np.int64)
    fs = estimate_fs(time_ns)

    print(f"Estimated fs: {fs:.2f} Hz")

    print("\n[STEP 3] Extracting signal...")
    signal = df[SIGNAL_COL].values.astype(np.float64)

    duration = len(signal) / fs
    print(f"Duration: {duration:.2f} seconds")

    print("\n[STEP 4] Writing EDF file...")

    channel_info = [{
        'label': 'ECG Lead II',
        'dimension': 'mV',
        'sample_frequency': int(fs),
        'physical_min': float(np.min(signal)),
        'physical_max': float(np.max(signal)),
        'digital_min': -32768,
        'digital_max': 32767,
        'transducer': '',
        'prefilter': ''
    }]

    writer = pyedflib.EdfWriter(
        out_edf,
        n_channels=1,
        file_type=pyedflib.FILETYPE_EDFPLUS
    )

    writer.setSignalHeaders(channel_info)
    writer.writeSamples([signal])
    writer.close()

    print("\n================ DONE =================")
    print(f"EDF file saved at: {out_edf}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_edf", type=str, required=True)

    args = parser.parse_args()

    convert_csv_to_edf(args.csv_path, args.out_edf)


if __name__ == "__main__":
    main()
