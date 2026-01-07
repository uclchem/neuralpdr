#!/bin/bash
# Organize and clean up messy files in the workspace
#
# Usage: ./util_cleanup.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "Workspace Cleanup"
echo "================================"

# Remove duplicate smooth_data.py from v3 directory
if [ -f "$PROJECT_ROOT/data/zenodo/v3/smooth_data.py" ]; then
    echo "  → Removing duplicate smooth_data.py from v3 directory..."
    rm "$PROJECT_ROOT/data/zenodo/v3/smooth_data.py"
    echo "  ✓ Removed data/zenodo/v3/smooth_data.py"
else
    echo "  ℹ️  No duplicate smooth_data.py found in v3 directory"
fi

# Check for and remove old processed files if they exist
if [ -d "$PROJECT_ROOT/data/processed" ]; then
    OLD_FILES=$(find "$PROJECT_ROOT/data/processed" -name "*_old*" -o -name "*_backup*" -o -name "*_temp*" 2>/dev/null || true)
    
    if [ -n "$OLD_FILES" ]; then
        echo "  → Removing old/backup/temp files from data/processed/..."
        echo "$OLD_FILES" | while read -r file; do
            if [ -f "$file" ]; then
                rm "$file"
                echo "  ✓ Removed: $(basename "$file")"
            fi
        done
    else
        echo "  ℹ️  No old/backup/temp files found in data/processed/"
    fi
fi

# Check for extracted v2 directories left behind
if [ -d "$PROJECT_ROOT/data/zenodo/v2/extracted" ]; then
    echo "  → Removing extracted v2 directory (should be cleaned by process script)..."
    rm -rf "$PROJECT_ROOT/data/zenodo/v2/extracted"
    echo "  ✓ Removed data/zenodo/v2/extracted/"
else
    echo "  ℹ️  No extracted v2 directory found"
fi

echo ""
echo "================================"
echo "✓ Workspace Cleanup Complete!"
echo "================================"
echo ""
echo "Your workspace is now organized:"
echo "  • All processing scripts in scripts/"
echo "  • Source data in data/zenodo/{v1,v2,v3}/"
echo "  • Processed data in data/processed/"
