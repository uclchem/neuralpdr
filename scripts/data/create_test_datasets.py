#!/usr/bin/env python
"""
Create minimal versions of v1, v2, v3 datasets for testing in CI.

This script extracts a small subset of models from each full dataset to create
lightweight test datasets that can be committed to the repository.

Usage:
    python create_test_datasets.py

Output:
    data/test/3dpdr_dataset_v1_test.h5  (~10 models, ~5 MB)
    data/test/3dpdr_dataset_v2_test.h5  (~10 models, ~5 MB)
    data/test/3dpdr_dataset_v3_test.h5  (~10 models, ~5 MB)
"""

import sys
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

# Paths
DATA_DIR = Path("data/processed")
TEST_DIR = Path("data/test")

# Number of models to include in test datasets
N_TEST_MODELS = 10

# Dataset configurations
DATASETS = {
    "v1": {
        "input": DATA_DIR / "3dpdr_dataset_v1.h5",
        "output": TEST_DIR / "3dpdr_dataset_v1_test.h5",
    },
    "v2": {
        "input": DATA_DIR / "3dpdr_dataset_v2.h5",
        "output": TEST_DIR / "3dpdr_dataset_v2_test.h5",
    },
    "v3": {
        "input": DATA_DIR / "3dpdr_dataset_v3.h5",
        "output": TEST_DIR / "3dpdr_dataset_v3_test.h5",
    },
}


def create_minimal_dataset(input_path: Path, output_path: Path, n_models: int = N_TEST_MODELS):
    """
    Create a minimal version of a dataset with only n_models.
    
    Args:
        input_path: Path to full dataset
        output_path: Path to output minimal dataset
        n_models: Number of models to extract
    """
    if not input_path.exists():
        print(f"⚠️  Input file not found: {input_path}")
        return False
    
    print(f"\nCreating minimal dataset from {input_path.name}")
    print(f"  Output: {output_path}")
    print(f"  Models: {n_models}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(input_path, "r") as f_in:
        # Get list of model keys (exclude metadata like 'header', 'species', etc.)
        metadata_keys = {"header", "species", "model_ids", "model_df"}
        all_keys = list(f_in.keys())
        model_keys = [k for k in all_keys if k not in metadata_keys]
        
        print(f"  Total models in source: {len(model_keys)}")
        
        # Select subset of models
        selected_models = model_keys[:n_models]
        
        with h5py.File(output_path, "w") as f_out:
            # Copy metadata
            for meta_key in metadata_keys:
                if meta_key in f_in:
                    data = f_in[meta_key][:]
                    
                    # For model_df and model_ids, only copy entries for selected models
                    if meta_key == "model_df" and len(model_keys) > 0:
                        # Assume model_df corresponds to model_keys order
                        data = data[:n_models]
                    elif meta_key == "model_ids" and len(model_keys) > 0:
                        # Get actual model IDs for selected models
                        if isinstance(data[0], bytes):
                            selected_ids = [mid for mid in data if mid.decode() in selected_models][:n_models]
                        else:
                            selected_ids = [mid for mid in data if mid in selected_models][:n_models]
                        data = np.array(selected_ids)
                    
                    f_out.create_dataset(meta_key, data=data, dtype=f_in[meta_key].dtype)
            
            # Copy selected models
            for model_key in tqdm(selected_models, desc="Copying models"):
                model_group = f_in[model_key]
                
                # Copy all datasets/groups in this model
                def copy_recursive(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        full_name = f"{model_key}/{name}" if name else model_key
                        f_out.create_dataset(full_name, data=obj[:], dtype=obj.dtype)
                    elif isinstance(obj, h5py.Group):
                        # Skip - groups are created automatically
                        pass
                
                # Copy model group contents
                if isinstance(model_group, h5py.Group):
                    model_group.visititems(copy_recursive)
                else:
                    # It's a dataset itself
                    f_out.create_dataset(model_key, data=model_group[:], dtype=model_group.dtype)
    
    # Check output file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Created: {size_mb:.2f} MB")
    
    return True


def main():
    """Create minimal test datasets for all versions."""
    print("=" * 60)
    print("Creating Minimal Test Datasets")
    print("=" * 60)
    
    success_count = 0
    
    for version, config in DATASETS.items():
        success = create_minimal_dataset(
            config["input"],
            config["output"],
            N_TEST_MODELS
        )
        if success:
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✓ Created {success_count}/{len(DATASETS)} test datasets")
    print("=" * 60)
    
    if success_count > 0:
        print("\nTest datasets can now be committed to the repository:")
        print("  git add data/test/*.h5")
        print("  git commit -m 'Add minimal test datasets for CI'")
        print("\nTo use in tests, update paths:")
        print("  V1_TEST_PATH = Path(__file__).parent.parent / 'data' / 'test' / '3dpdr_dataset_v1_test.h5'")


if __name__ == "__main__":
    main()
