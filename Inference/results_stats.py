import os
import pandas as pd

# =========================
# INPUT
# =========================
RESULTS_DIR = "../Results_new"   # folder containing patient CSVs
OUTPUT_CSV = "patient_summary_new_2.csv"

summary_rows = []

# =========================
# LOOP OVER ALL PATIENT FILES
# =========================
for file in os.listdir(RESULTS_DIR):
    if not file.endswith(".csv"):
        continue

    patient_name = os.path.splitext(file)[0]
    csv_path = os.path.join(RESULTS_DIR, file)

    try:
        df = pd.read_csv(csv_path)

        if "pred_label" not in df.columns:
            print(f"[SKIP] No pred_label in {file}")
            continue

        # Count labels
        af_count = (df["pred_label"] == "AF").sum()
        notaf_count = (df["pred_label"] == "NotAF").sum()

        total = len(df)

        summary_rows.append({
            "patient": patient_name,
            "AF_count": af_count,
            "NotAF_count": notaf_count,
            "total_segments": total,
            "AF_ratio": af_count / total if total > 0 else 0
        })

    except Exception as e:
        print(f"[ERROR] {file}: {e}")

# =========================
# SAVE RESULT
# =========================
summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values("AF_count", ascending=False)

summary_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n[SAVED] {OUTPUT_CSV}")
print(summary_df.head())
