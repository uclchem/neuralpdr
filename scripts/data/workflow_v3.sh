#!/bin/bash
# Process v3 dataset: copy to processed/ and apply smoothing
#
# Usage: ./workflow_v3.sh
#
# This script:
# 1. Copies v3 HDF5 to processed/ directory
# 2. Applies required smoothing to create smoothed version

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"  # Up two levels: data/ then scripts/

# Paths
SOURCE_FILE="$PROJECT_ROOT/data/zenodo/v3/3dpdr_dataset_v3.h5"
OUTPUT_FILE="$PROJECT_ROOT/data/processed/3dpdr_dataset_v3.h5"
SMOOTH_FILE="$PROJECT_ROOT/data/processed/3dpdr_dataset_v3_smooth.h5"

echo "================================"
echo "Processing v3 Dataset"
echo "================================"

# Check if source exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "ERROR: Source file not found at $SOURCE_FILE"
    exit 1
fi

# Step 1: Copy to processed directory
echo ""
echo "Step 1: Copying to processed directory..."
mkdir -p "$PROJECT_ROOT/data/processed"
cp "$SOURCE_FILE" "$OUTPUT_FILE"
echo "  ✓ Copied to: $OUTPUT_FILE"

SOURCE_SIZE=$(du -h "$SOURCE_FILE" | cut -f1)
echo "  ✓ File size: $SOURCE_SIZE"

# Step 2: Apply smoothing (REQUIRED)
echo ""
echo "Step 2: Applying smoothing (this may take a while)..."
python "$SCRIPT_DIR/v3_smooth.py" "$OUTPUT_FILE"

# Verify smoothed file was created
if [ -f "$SMOOTH_FILE" ]; then
    SMOOTH_SIZE=$(du -h "$SMOOTH_FILE" | cut -f1)
    echo "  ✓ Created smoothed version: $SMOOTH_FILE"
    echo "  ✓ Smoothed file size: $SMOOTH_SIZE"
else
    echo "ERROR: Smoothed file not created!"
    exit 1
fi

# Final verification
echo ""
echo "================================"
echo "✓ v3 Processing Complete!"
echo "================================"
echo "Raw file:      $OUTPUT_FILE"
echo "Smoothed file: $SMOOTH_FILE"
