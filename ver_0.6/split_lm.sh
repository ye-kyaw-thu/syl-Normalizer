#!/bin/bash

# Configuration
INPUT_FILE="myMono_syl_trigram.arpa"
OUTPUT_DIR="lm_chunks"
CHUNK_SIZE="24M"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Splitting $INPUT_FILE into $OUTPUT_DIR in ${CHUNK_SIZE} chunks..."

# Split the file with numeric suffixes
# This will create files like myMono_syl_trigram.arpa.part-01, part-02, etc.
split -b "$CHUNK_SIZE" --numeric-suffixes=1 "$INPUT_FILE" "$OUTPUT_DIR/${INPUT_FILE}.part-"

# Generate a checksum for verification
sha256sum "$INPUT_FILE" > "$OUTPUT_DIR/checksum.sha256"

echo "Done. Created $(ls "$OUTPUT_DIR" | wc -l | xargs -n1 expr -1) parts in '$OUTPUT_DIR'."

