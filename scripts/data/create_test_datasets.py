#!/usr/bin/env python
"""
Create minimal versions of v1, v2, v3, v4 datasets for testing in CI.

This script extracts a small subset of models from each full dataset to create
lightweight test datasets that can be committed to the repository.

Note: In this codebase, 1 model (HDF5 group) = 1 training sample.
Each model represents one simulation run.

Usage:
    python create_test_datasets.py                    # Create with 10 models (default)
    python create_test_datasets.py --n-models 128     # Create with 128 models
    python create_test_datasets.py --n-models 5       # Create with 5 models
    python create_test_datasets.py --skip-extra-data  # Skip species, keep header
    python create_test_datasets.py --n-models 128 --skip-extra-data --enable-compression  # Minimal size

Output:
    data/test/3dpdr_dataset_v1_test.h5
    data/test/3dpdr_dataset_v2_test.h5
    data/test/3dpdr_dataset_v3_test.h5
    data/test/3dpdr_dataset_v4_test.h5
"""

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Paths
DATA_DIR = Path("data/processed")
TEST_DIR = Path("data/test")

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
    "v4": {
        "input": DATA_DIR / "3dpdr_dataset_v4.h5",
        "output": TEST_DIR / "3dpdr_dataset_v4_test.h5",
    },
}


def select_models(model_keys, n_models):
    """Randomly sample up to n_models keys.

    A first-N slice would bias v4's test fixture toward axis-aligned models
    (written first) and, within those, toward one corner of the grid - random
    sampling over the full key list avoids that for free, without needing to
    know anything about how a given dataset version names or orders its models.
    Seeded for reproducible test fixtures across regenerations.
    """
    if n_models >= len(model_keys):
        return model_keys
    random.seed(0)
    return random.sample(model_keys, n_models)


def create_minimal_dataset(
    input_path: Path,
    output_path: Path,
    n_models: int,
    skip_extra_data: bool = False,
    enable_compression: bool = False,
):
    """
    Create a minimal version of a dataset with only n_models.

    Args:
        input_path: Path to full dataset
        output_path: Path to output minimal dataset
        n_models: Number of models (training samples) to extract
        skip_extra_data: If True, skip species and only copy pdr data from models (keeps header, model_ids, model_df)
        enable_compression: If True, enable gzip compression for all datasets
    """
    if not input_path.exists():
        print(f"⚠️  Input file not found: {input_path}")
        return False

    print(f"\nCreating minimal dataset from {input_path.name}")
    print(f"  Output: {output_path}")
    print(f"  Models: {n_models}")
    if skip_extra_data:
        print("  Mode: PDR data + header, model_ids, model_df only (skip species)")
    if enable_compression:
        print("  Compression: Enabled (gzip)")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as f_in:
        # Get list of model keys (exclude metadata like 'header', 'species', etc.)
        metadata_keys = {"header", "species", "model_ids", "model_df"}
        all_keys = list(f_in.keys())
        model_keys = [k for k in all_keys if k not in metadata_keys]

        print(f"  Total models in source: {len(model_keys)}")

        # Select n_models, balanced across model "families" if more than one is present
        selected_models = select_models(model_keys, n_models)
        print(f"  Selected {len(selected_models)} models")

        with h5py.File(output_path, "w") as f_out:
            # Copy metadata
            # Always copy model_ids and model_df for selected models
            # Skip species if skip_extra_data is True (header is required by data loader)
            keys_to_copy = (
                metadata_keys
                if not skip_extra_data
                else {"model_ids", "model_df", "header", "species"}
            )

            for meta_key in keys_to_copy:
                if meta_key in f_in:
                    data = f_in[meta_key][:]

                    # For model_df and model_ids, only copy entries for selected models
                    if meta_key == "model_df" and len(model_keys) > 0:
                        # Assume model_df corresponds to model_keys order
                        data = data[:n_models]
                    elif meta_key == "model_ids" and len(model_keys) > 0:
                        # Get actual model IDs for selected models
                        if isinstance(data[0], bytes):
                            selected_ids = [
                                mid for mid in data if mid.decode() in selected_models
                            ][:n_models]
                        else:
                            selected_ids = [
                                mid for mid in data if mid in selected_models
                            ][:n_models]
                        data = np.array(selected_ids)

                    f_out.create_dataset(
                        meta_key,
                        data=data,
                        dtype=f_in[meta_key].dtype,
                        compression="gzip" if enable_compression else None,
                        compression_opts=9 if enable_compression else None,
                    )

            # Copy selected models
            for model_key in tqdm(selected_models, desc="Copying models"):
                model_group = f_in[model_key]

                if (
                    skip_extra_data
                    and isinstance(model_group, h5py.Group)
                    and "pdr" in model_group
                ):
                    # Only copy the pdr subdirectory
                    pdr_group = model_group["pdr"]

                    if isinstance(pdr_group, h5py.Dataset):
                        # pdr is a dataset itself
                        f_out.create_dataset(
                            f"{model_key}/pdr",
                            data=pdr_group[:],
                            dtype=pdr_group.dtype,
                            compression="gzip" if enable_compression else None,
                            compression_opts=9 if enable_compression else None,
                        )
                    else:
                        # pdr is a group with sub-datasets
                        def copy_pdr_recursive(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                full_name = (
                                    f"{model_key}/pdr/{name}"
                                    if name
                                    else f"{model_key}/pdr"
                                )
                                f_out.create_dataset(
                                    full_name,
                                    data=obj[:],
                                    dtype=obj.dtype,
                                    compression="gzip" if enable_compression else None,
                                    compression_opts=9 if enable_compression else None,
                                )
                            elif isinstance(obj, h5py.Group):
                                pass  # Groups are created automatically

                        pdr_group.visititems(copy_pdr_recursive)
                else:
                    # Copy all datasets/groups in this model
                    def copy_recursive(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            full_name = f"{model_key}/{name}" if name else model_key
                            f_out.create_dataset(
                                full_name,
                                data=obj[:],
                                dtype=obj.dtype,
                                compression="gzip" if enable_compression else None,
                                compression_opts=9 if enable_compression else None,
                            )
                        elif isinstance(obj, h5py.Group):
                            # Skip - groups are created automatically
                            pass

                    # Copy model group contents
                    if isinstance(model_group, h5py.Group):
                        model_group.visititems(copy_recursive)
                    else:
                        # It's a dataset itself
                        f_out.create_dataset(
                            model_key,
                            data=model_group[:],
                            dtype=model_group.dtype,
                            compression="gzip" if enable_compression else None,
                            compression_opts=9 if enable_compression else None,
                        )

    # Check output file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Created: {size_mb:.2f} MB")

    return True


def main():
    """Create minimal test datasets for all versions."""
    parser = argparse.ArgumentParser(
        description="Create minimal test datasets from full datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=10,
        help="Number of models to include in test datasets",
    )
    parser.add_argument(
        "--skip-extra-data",
        action="store_true",
        help="Skip species, only copy pdr data from models (keeps header, model_ids, model_df)",
    )
    parser.add_argument(
        "--enable-compression",
        action="store_true",
        help="Enable gzip compression (level 9) for all datasets to minimize file size",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Creating Minimal Test Datasets")
    print("=" * 60)
    print(f"Number of models: {args.n_models}")
    if args.skip_extra_data:
        print("Mode: PDR data only (skip species, keep header, model_ids, model_df)")
    if args.enable_compression:
        print("Compression: Enabled (gzip level 9)")
    print("=" * 60)

    success_count = 0

    # Create test datasets
    for version, config in DATASETS.items():
        success = create_minimal_dataset(
            config["input"],
            config["output"],
            args.n_models,
            args.skip_extra_data,
            args.enable_compression,
        )
        if success:
            success_count += 1

    print("\n" + "=" * 60)
    print(f"✓ Created {success_count}/{len(DATASETS)} test datasets")
    print("=" * 60)

    if success_count > 0:
        print("\nTest datasets created:")
        print(f"  {args.n_models} models per dataset: data/test/*_test.h5")
        print("\nThese can be committed to the repository:")
        print("  git add data/test/*.h5")
        print("  git commit -m 'Add test datasets for CI'")
        print("\nTo use in tests:")
        print("  dataset_path = 'data/test/3dpdr_dataset_v3_test.h5'")
        print("\nTo create different sizes:")
        print(
            "  python create_test_datasets.py --n-models 128  # For ultra-fast testing"
        )
        print("  python create_test_datasets.py --n-models 5    # For minimal testing")
        print("\nTo create minimal datasets (PDR data only):")
        print("  python create_test_datasets.py --n-models 128 --skip-extra-data")
        print(
            "  python create_test_datasets.py --n-models 128 --skip-extra-data --enable-compression  # Smallest size"
        )


if __name__ == "__main__":
    main()
