#!/bin/bash
# Process v4 dataset: raw 3D-PDR grid (COL128.*) -> HDF5 sightlines with headers
#
# Usage: ./workflow_v4.sh
#
# This script:
# 1. Converts data/v4/COL128.* into per-sightline HDF5 models (axis-aligned +
#    ray-marched diagonal), via v4_raw_to_h5.py
# 2. Adds header/species/model_ids datasets, via add_file_headers_v4.py
#
# Unlike v3, v4 needs no separate smoothing step: both the axis-aligned and
# diagonal sightlines are monotonic in visual_extinction by construction.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"  # Up two levels: data/ then scripts/

# Paths
RAW_DIR="$PROJECT_ROOT/data/v4"
OUTPUT_FILE="$PROJECT_ROOT/data/processed/3dpdr_dataset_v4.h5"

echo "================================"
echo "Processing v4 Dataset"
echo "================================"

# Check if source exists
if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: Raw data directory not found at $RAW_DIR"
    exit 1
fi

mkdir -p "$PROJECT_ROOT/data/processed"

# Step 1: raw grid -> HDF5 sightlines
echo ""
echo "Step 1: Converting raw grid to sightlines (this may take several minutes)..."
python "$SCRIPT_DIR/v4_raw_to_h5.py" "$RAW_DIR" "$OUTPUT_FILE"

# Step 2: add header/species/model_ids datasets
echo ""
echo "Step 2: Adding headers..."
python "$SCRIPT_DIR/add_file_headers_v4.py" "$OUTPUT_FILE" "model_(axis|diag)_"

# Final verification
if [ -f "$OUTPUT_FILE" ]; then
    OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo ""
    echo "================================"
    echo "✓ v4 Processing Complete!"
    echo "================================"
    echo "Output file: $OUTPUT_FILE"
    echo "File size:   $OUTPUT_SIZE"
else
    echo "ERROR: Output file not created!"
    exit 1
fi
