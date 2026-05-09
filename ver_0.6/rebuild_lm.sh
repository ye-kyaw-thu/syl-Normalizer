#!/bin/bash

# Configuration
INPUT_DIR="lm_chunks"
OUTPUT_FILE="myMono_syl_trigram.arpa"

echo "Reassembling $OUTPUT_FILE..."

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory $INPUT_DIR not found."
    exit 1
fi

# Combine all parts back together
cat "$INPUT_DIR/${OUTPUT_FILE}.part-"* > "$OUTPUT_FILE"

# Verify integrity using the checksum file
if [ -f "$INPUT_DIR/checksum.sha256" ]; then
    echo "Verifying file integrity..."
    if sha256sum --status -c "$INPUT_DIR/checksum.sha256"; then
        echo "Success: $OUTPUT_FILE is identical to the original."
    else
        echo "Error: Checksum failed. The file may be incomplete."
    fi
else
    echo "Warning: No checksum file found in $INPUT_DIR. Verification skipped."
fi

