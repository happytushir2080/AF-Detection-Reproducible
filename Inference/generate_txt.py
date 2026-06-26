import os
import argparse

def find_csvs(root_dir, out_txt, log_txt):
    patients = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    print(f"[INFO] Found {len(patients)} patients")

    valid_entries = []
    logs = []

    for patient in patients:
        patient_path = os.path.join(root_dir, patient)

        for sub in ["1", "2", "3"]:
            sub_path = os.path.join(patient_path, sub)

            if not os.path.exists(sub_path):
                logs.append(f"[MISSING] {patient} -> folder '{sub}' not found")
                continue

            # Check Holter / HOLTER
            holter_path = None
            for h in ["Holter", "HOLTER"]:
                temp = os.path.join(sub_path, h)
                if os.path.exists(temp):
                    holter_path = temp
                    break

            if holter_path is None:
                logs.append(f"[MISSING] {patient}/{sub} -> Holter folder not found")
                continue

            # Find CSV files
            csv_files = [f for f in os.listdir(holter_path) if f.endswith(".csv")]

            if len(csv_files) == 0:
                logs.append(f"[MISSING] {patient}/{sub}/Holter -> No CSV found")
                continue

            # Take all CSVs (or first one if needed)
            for csv_file in csv_files:
                csv_path = os.path.join(holter_path, csv_file)

                tag = f"{patient}_{sub}"
                valid_entries.append(f"{csv_path},{tag}")

    # -------------------------
    # Save TXT
    # -------------------------
    with open(out_txt, "w") as f:
        for line in valid_entries:
            f.write(line + "\n")

    print(f"[SAVED] CSV list → {out_txt}")
    print(f"[INFO] Total valid entries: {len(valid_entries)}")

    # -------------------------
    # Save LOG
    # -------------------------
    with open(log_txt, "w") as f:
        for log in logs:
            f.write(log + "\n")

    print(f"[SAVED] Logs → {log_txt}")
    print(f"[INFO] Total issues logged: {len(logs)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing patient folders")
    parser.add_argument("--out_txt", type=str, default="csv_list.txt")
    parser.add_argument("--log_txt", type=str, default="missing_log.txt")

    args = parser.parse_args()

    find_csvs(args.root_dir, args.out_txt, args.log_txt)


if __name__ == "__main__":
    main()

# python generate_txt.py \
#   --root_dir /mnt/project2/DST-SERB/AIIMS-data \
#   --out_txt csv_list.txt \
#   --log_txt missing_log.txt