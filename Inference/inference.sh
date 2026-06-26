#!/bin/bash

INPUT_DIR="../new_recordings"
OUTPUT_DIR="../Results_new"
LOG_DIR="../logs_new"

CHECKPOINT="../code-7_patient_split/best_spec_crnn.pt"

BATCH_SIZE=32
THRESHOLD=0.9891

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=========== STARTING BATCH INFERENCE ==========="

# Loop over each patient folder
for patient_path in "$INPUT_DIR"/*
do
    # Skip if not a directory
    [ -d "$patient_path" ] || continue

    patient_name=$(basename "$patient_path")

    echo "----------------------------------------"
    echo "Processing patient: $patient_name"

    output_csv="$OUTPUT_DIR/${patient_name}.csv"
    log_file="$LOG_DIR/${patient_name}.log"

    echo "Saving output to: $output_csv"
    echo "Logging to: $log_file"

    python inference.py \
        --input_dir "$patient_path" \
        --checkpoint "$CHECKPOINT" \
        --output_csv "$output_csv" \
        --batch_size $BATCH_SIZE \
        --threshold $THRESHOLD \
        --use_notch \
        > "$log_file" 2>&1

    echo "Done: $patient_name"
done

echo "=========== ALL DONE ==========="
