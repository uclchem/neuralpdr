#!/bin/bash
# Process v2 dataset from tarball to final HDF5 with headers
#
# Usage: ./workflow_v2.sh [options]
#
# Options:
#   --metallicity <Z>: Which metallicity to process (Z0p1, Z0p5, Z1p0, Z2p0)
#                      Default: Z1p0 (solar metallicity)
#   --include-metallicity: Add metallicity value as first column in time series
#   --all-metallicities: Process ALL metallicities with appended dataset names
#
# This script:
# 1. Extracts the v2 tarball
# 2. Converts .params/.fin files to HDF5 (Z1p0 by default or all)
# 3. Adds proper headers
# 4. Cleans up temporary files

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"  # Up two levels: data/ then scripts/

# Default values
METALLICITY="Z1p0"
INCLUDE_METALLICITY_FLAG=""
ALL_METALLICITIES_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metallicity)
            METALLICITY="$2"
            shift 2
            ;;
        --include-metallicity)
            INCLUDE_METALLICITY_FLAG="--include-metallicity"
            shift
            ;;
        --all-metallicities)
            ALL_METALLICITIES_FLAG="--all-metallicities"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./workflow_v2.sh [--metallicity Z1p0] [--include-metallicity] [--all-metallicities]"
            exit 1
            ;;
    esac
done

# Paths
TARBALL="$PROJECT_ROOT/data/zenodo/v2/simulations.tgz"
EXTRACT_DIR="$PROJECT_ROOT/data/zenodo/v2/extracted"
RAW_OUTPUT="$PROJECT_ROOT/data/processed/3dpdr_dataset_v2_raw.h5"
FINAL_OUTPUT="$PROJECT_ROOT/data/processed/3dpdr_dataset_v2.h5"

echo "================================"
echo "Processing v2 Dataset"
echo "================================"

if [ -n "$ALL_METALLICITIES_FLAG" ]; then
    echo "Mode: ALL metallicities (Z0p1, Z0p5, Z1p0, Z2p0)"
else
    echo "Mode: Single metallicity ($METALLICITY)"
fi

if [ -n "$INCLUDE_METALLICITY_FLAG" ]; then
    echo "Metallicity column: Enabled"
fi
echo ""

# Check if tarball exists
if [ ! -f "$TARBALL" ]; then
    echo "ERROR: Tarball not found at $TARBALL"
    exit 1
fi

# Step 1: Extract tarball
echo "Step 1: Extracting tarball..."
mkdir -p "$EXTRACT_DIR"
tar -xzf "$TARBALL" -C "$EXTRACT_DIR"

# Check if metallicity directory exists
METAL_DIR="$EXTRACT_DIR/$METALLICITY"
if [ ! -d "$METAL_DIR" ]; then
    echo "ERROR: Metallicity directory not found: $METAL_DIR"
    echo "Available metallicities:"
    ls -d "$EXTRACT_DIR"/Z*/ 2>/dev/null || echo "  None found"
    exit 1
fi

echo "  ✓ Extracted to: $EXTRACT_DIR"
echo "  ✓ Using metallicity: $METALLICITY"

# Count files
PARAMS_COUNT=$(find "$METAL_DIR" -name "*.params" | wc -l | tr -d ' ')
echo "  ✓ Found $PARAMS_COUNT models"

# Step 2: Convert to HDF5
echo ""
echo "Step 2: Converting to HDF5..."
python "$SCRIPT_DIR/v2_raw_to_h5.py" "$EXTRACT_DIR" "$RAW_OUTPUT" --metallicity "$METALLICITY" $INCLUDE_METALLICITY_FLAG $ALL_METALLICITIES_FLAG

# Step 3: Add headers
echo ""
echo "Step 3: Adding headers..."
python "$SCRIPT_DIR/v2_add_headers.py" "$RAW_OUTPUT" "$FINAL_OUTPUT"

# Step 4: Cleanup
echo ""
echo "Step 4: Cleaning up temporary files..."
rm -f "$RAW_OUTPUT"
rm -rf "$EXTRACT_DIR"
echo "  ✓ Removed temporary extraction directory"
echo "  ✓ Removed raw HDF5 file"

# Final verification
echo ""
echo "================================"
echo "✓ v2 Processing Complete!"
echo "================================"
echo "Final output: $FINAL_OUTPUT"

if [ -f "$FINAL_OUTPUT" ]; then
    SIZE=$(du -h "$FINAL_OUTPUT" | cut -f1)
    echo "File size: $SIZE"
else
    echo "ERROR: Final output file not created!"
    exit 1
fi
