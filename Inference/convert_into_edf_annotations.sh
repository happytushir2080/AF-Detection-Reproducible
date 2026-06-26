#!/bin/bash

SCRIPT="convert_into_edf_annotations.py"   # your python file name
INPUT_DIR="$1"
BASE_OUT_DIR="$2"

if [ -z "$INPUT_DIR" ] || [ -z "$BASE_OUT_DIR" ]; then
  echo "Usage: ./run_annotations.sh input_dir output_dir"
  exit 1
fi

echo "===== STARTING ANNOTATION GENERATION ====="
echo "Input Dir  : $INPUT_DIR"
echo "Output Dir : $BASE_OUT_DIR"

# create base output directory
mkdir -p "$BASE_OUT_DIR"

# loop over all csv files
find "$INPUT_DIR" -type f -name "*.csv" | while read -r csv_path
do
  echo "----------------------------------------"
  echo "Processing: $csv_path"

  # extract filename without extension
  filename=$(basename "$csv_path" .csv)

  # output txt path
  out_txt="$BASE_OUT_DIR/$filename.txt"

  echo "Output TXT: $out_txt"

  python $SCRIPT \
    --pred_csv "$csv_path" \
    --out_txt "$out_txt"

  echo "DONE: $csv_path"

done

echo "========== ALL DONE =========="

# ./convert_into_edf_annotations.sh Results_new/ convert_full_record_into_edf/annotations
