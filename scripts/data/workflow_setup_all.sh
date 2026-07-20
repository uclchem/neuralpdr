#!/bin/bash
# Master script to process all three datasets (v1, v2, v3)
#
# Usage: ./workflow_setup_all.sh [options]
#
# Options:
#   --skip-v1    Skip v1 processing
#   --skip-v2    Skip v2 processing
#   --skip-v3    Skip v3 processing
#   --skip-v4    Skip v4 processing
#   --help       Show this help message

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"  # Up two levels: data/ then scripts/

# Parse arguments
SKIP_V1=false
SKIP_V2=false
SKIP_V3=false
SKIP_V4=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-v1)
            SKIP_V1=true
            shift
            ;;
        --skip-v2)
            SKIP_V2=true
            shift
            ;;
        --skip-v3)
            SKIP_V3=true
            shift
            ;;
        --skip-v4)
            SKIP_V4=true
            shift
            ;;
        --help)
            echo "Usage: ./workflow_setup_all.sh [options]"
            echo ""
            echo "Options:"
            echo "  --skip-v1    Skip v1 processing"
            echo "  --skip-v2    Skip v2 processing"
            echo "  --skip-v3    Skip v3 processing"
            echo "  --skip-v4    Skip v4 processing"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "NeuralPDR Dataset Processing Pipeline"
echo "========================================"
echo ""
echo "This will process all datasets from data/zenodo/"
echo "and create final versions in data/processed/"
echo ""

# Create processed directory
mkdir -p "$PROJECT_ROOT/data/processed"

# Process v1
if [ "$SKIP_V1" = false ]; then
    echo ""
    echo "╔════════════════════════════════╗"
    echo "║  Processing v1 Dataset         ║"
    echo "╚════════════════════════════════╝"
    
    python "$SCRIPT_DIR/v1_add_headers.py" \
        "$PROJECT_ROOT/data/zenodo/v1/3dpdr_dataset_8192.h5" \
        "$PROJECT_ROOT/data/processed/3dpdr_dataset_v1.h5"
else
    echo ""
    echo "⏭️  Skipping v1 processing"
fi

# Process v2
if [ "$SKIP_V2" = false ]; then
    echo ""
    echo "╔════════════════════════════════╗"
    echo "║  Processing v2 Dataset         ║"
    echo "╚════════════════════════════════╝"
    
    chmod +x "$SCRIPT_DIR/workflow_v2.sh"
    "$SCRIPT_DIR/workflow_v2.sh"
else
    echo ""
    echo "⏭️  Skipping v2 processing"
fi

# Process v3
if [ "$SKIP_V3" = false ]; then
    echo ""
    echo "╔════════════════════════════════╗"
    echo "║  Processing v3 Dataset         ║"
    echo "╚════════════════════════════════╝"
    
    chmod +x "$SCRIPT_DIR/workflow_v3.sh"
    "$SCRIPT_DIR/workflow_v3.sh"
else
    echo ""
    echo "⏭️  Skipping v3 processing"
fi

# Process v4
if [ "$SKIP_V4" = false ]; then
    echo ""
    echo "╔════════════════════════════════╗"
    echo "║  Processing v4 Dataset         ║"
    echo "╚════════════════════════════════╝"

    chmod +x "$SCRIPT_DIR/workflow_v4.sh"
    "$SCRIPT_DIR/workflow_v4.sh"
else
    echo ""
    echo "⏭️  Skipping v4 processing"
fi

# Summary
echo ""
echo "========================================"
echo "✓ All Dataset Processing Complete!"
echo "========================================"
echo ""
echo "Processed files in data/processed/:"
ls -lh "$PROJECT_ROOT/data/processed/" | tail -n +2

echo ""
echo "You can now use these datasets with the neuralpdr package!"
