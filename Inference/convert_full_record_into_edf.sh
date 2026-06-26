#!/bin/bash

SCRIPT="convert_full_record_into_edf.py"
INPUT_FILE="$1"
BASE_OUT_DIR="$2"

if [ -z "$INPUT_FILE" ] || [ -z "$BASE_OUT_DIR" ]; then
  echo "Usage: ./run_csv_to_edf.sh csv_list.txt base_output_dir"
  exit 1
fi

echo "===== STARTING CSV → EDF BATCH ====="
echo "Base output dir: $BASE_OUT_DIR"

while IFS=',' read -r csv_path out_name
do
  # skip empty lines
  if [ -z "$csv_path" ] || [ -z "$out_name" ]; then
    echo "Skipping invalid line"
    continue
  fi

  # trim spaces
  csv_path=$(echo "$csv_path" | xargs)
  out_name=$(echo "$out_name" | xargs)

  echo "----------------------------------------"
  echo "CSV Path : $csv_path"
  echo "Out Name : $out_name"

  # construct output path
  out_edf="$BASE_OUT_DIR/$out_name.edf"

  echo "Output EDF : $out_edf"

  # create base dir (only once is fine, but safe here)
  mkdir -p "$BASE_OUT_DIR"

  python $SCRIPT \
    --csv_path "$csv_path" \
    --out_edf "$out_edf"

  echo "DONE: $csv_path"

done < "$INPUT_FILE"

echo "========== ALL DONE =========="

#./convert_full_record_into_edf.sh csv_list_NEW_RECORDING.txt ./convert_full_record_into_edf/records
