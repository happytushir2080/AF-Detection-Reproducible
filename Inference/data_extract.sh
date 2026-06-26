#!/bin/bash

SCRIPT="single_patient_data_split.py"   # your python script

INPUT_FILE="$1"

if [ -z "$INPUT_FILE" ]; then
  echo "Usage: ./run_batch.sh jobs.txt"
  exit 1
fi

echo "Starting batch processing..."

while IFS=',' read -r csv_path out_dir
do
  echo "----------------------------------------"
  echo "Processing: $csv_path"

  # 👉 add prefix here
  final_out_dir="updated_results/new_recordings/$out_dir"

  echo "Output Dir: $final_out_dir"

  # 👉 create directory (important)
  mkdir -p "$final_out_dir"

  python $SCRIPT \
    --csv_path "$csv_path" \
    --out_dir "$final_out_dir"

  echo "Done: $csv_path"
done < "$INPUT_FILE"



echo "========== ALL DONE =========="
