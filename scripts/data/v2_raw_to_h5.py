#!/usr/bin/env python
"""
Process v2 dataset from raw .params and .fin files to HDF5 format.

By default, processes only Z1p0 (normal/solar metallicity) data.

Usage:
    python models_to_h5_v2.py <data_path> <output_file> [options]

Arguments:
    data_path: Path to directory containing metallicity subdirectories (Z0p1/, Z1p0/, etc.)
    output_file: Path to output HDF5 file
    --metallicity: Which metallicity to process (default: Z1p0)
                   Options: Z0p1, Z0p5, Z1p0 (default), Z2p0
    --include-metallicity: Add metallicity value as first column in time series data
    --all-metallicities: Process ALL metallicities with appended dataset names
                         (overrides --metallicity flag)

Example:
    # Default: Z1p0 only
    python models_to_h5_v2.py data/zenodo/v2/extracted data/processed/3dpdr_dataset_v2_raw.h5

    # Specific metallicity with metallicity column
    python models_to_h5_v2.py data/zenodo/v2/extracted data/processed/3dpdr_dataset_v2_raw.h5 --metallicity Z0p5 --include-metallicity
    
    # All metallicities in one file
    python models_to_h5_v2.py data/zenodo/v2/extracted data/processed/3dpdr_dataset_v2_raw.h5 --all-metallicities
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

SPECIES = [
    "H3+",
    "He+",
    "Mg",
    "H2+",
    "O2",
    "CH5+",
    "CH4+",
    "O+",
    "OH+",
    "Mg+",
    "C+",
    "CH4",
    "H2O+",
    "H3O+",
    "CO+",
    "O2+",
    "CH2",
    "H2O",
    "H+",
    "CH3+",
    "CH",
    "CH3",
    "HCO+",
    "CH2+",
    "C",
    "He",
    "CH+",
    "CO",
    "OH",
    "O",
    "H2",
    "H",
    "e-",
]

METALLICITY_VALUES = {"Z0p1": 0.1, "Z0p5": 0.5, "Z1p0": 1.0, "Z2p0": 2.0}


def process_all_metallicities(
    data_path: str, output_file: str, include_metallicity: bool = False
):
    """
    Process all metallicities into a single HDF5 file with appended dataset names.

    Args:
        data_path: Path to directory containing metallicity subdirectories
        output_file: Path to output HDF5 file
        include_metallicity: Whether to prepend metallicity value to time series
    """
    data_path = Path(data_path)
    output_file = Path(output_file)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Processing v2 dataset (ALL METALLICITIES):")
    print(f"  Data path: {data_path}")
    print(f"  Include metallicity column: {include_metallicity}")
    print(f"  Output file: {output_file}")
    print(f"  Metallicities: {', '.join(METALLICITY_VALUES.keys())}")

    all_model_ids = []
    all_params = []
    total_models = 0

    with h5py.File(output_file, "w") as fh:
        print("\nProcessing metallicities...")

        for metallicity in sorted(METALLICITY_VALUES.keys()):
            metallicity_dir = data_path / metallicity
            metallicity_value = METALLICITY_VALUES[metallicity]

            if not metallicity_dir.exists():
                print(f"  ⚠️  Skipping {metallicity} - directory not found")
                continue

            # Find all .params files
            params_files = sorted(metallicity_dir.glob("*.params"))
            model_ids = [p.stem for p in params_files]
            n_models = len(model_ids)

            print(
                f"\n  Processing {metallicity} (Z = {metallicity_value}): {n_models} models"
            )

            if n_models == 0:
                print("    ⚠️  No .params files found, skipping")
                continue

            for model_id in tqdm(model_ids, desc=f"  {metallicity}", leave=False):
                # Append metallicity to model_id
                full_model_id = f"{model_id}_{metallicity}"
                all_model_ids.append(full_model_id)

                # Read parameters
                params = np.genfromtxt(metallicity_dir / f"{model_id}.params")
                all_params.append(params)

                # Read data files
                for datatype in ["pdr", "spop"]:
                    fin_file = metallicity_dir / f"{model_id}.{datatype}.fin"

                    if fin_file.exists():
                        pdr_data = np.genfromtxt(fin_file)

                        # Prepend metallicity column if requested
                        if include_metallicity:
                            n_rows = pdr_data.shape[0]
                            metallicity_col = np.full(
                                (n_rows, 1), metallicity_value, dtype=np.float32
                            )
                            pdr_data = np.hstack([metallicity_col, pdr_data])

                        fh.create_dataset(
                            name=f"{full_model_id}/{datatype}",
                            data=pdr_data,
                            dtype="float32",
                        )

            total_models += n_models

        # Store metadata
        fh.create_dataset("model_df", data=np.array(all_params), dtype="float32")
        fh.create_dataset("model_ids", data=all_model_ids, dtype="S20")  # Longer for suffix
        fh.create_dataset("species", data=SPECIES, dtype="S10")

        # Store metallicity info
        fh.attrs["metallicity"] = "all"
        fh.attrs["metallicities_included"] = ", ".join(
            sorted(METALLICITY_VALUES.keys())
        )
        fh.attrs["includes_metallicity_column"] = include_metallicity

    print(f"\n✓ Successfully created {output_file}")
    print(f"  - Metallicities: {', '.join(sorted(METALLICITY_VALUES.keys()))}")
    print(f"  - Total models: {total_models}")
    print(f"  - Species: {len(SPECIES)}")
    print(f"  - Metallicity column: {'Yes' if include_metallicity else 'No'}")


def process_v2_dataset(
    data_path: str,
    output_file: str,
    metallicity: str = "Z1p0",
    include_metallicity: bool = False,
):
    """
    Process v2 dataset from .params and .fin files to HDF5.

    Args:
        data_path: Path to directory containing metallicity subdirectories
        output_file: Path to output HDF5 file
        metallicity: Which metallicity to process (Z0p1, Z0p5, Z1p0, Z2p0)
        include_metallicity: Whether to prepend metallicity value to time series
    """
    data_path = Path(data_path)
    output_file = Path(output_file)

    # Validate metallicity
    if metallicity not in METALLICITY_VALUES:
        print(
            f"ERROR: Invalid metallicity '{metallicity}'. Must be one of: {', '.join(METALLICITY_VALUES.keys())}"
        )
        sys.exit(1)

    # Find the metallicity subdirectory
    metallicity_dir = data_path / metallicity

    if not metallicity_dir.exists():
        print(f"ERROR: Metallicity directory not found: {metallicity_dir}")
        print(f"Available directories in {data_path}:")
        for d in data_path.iterdir():
            if d.is_dir():
                print(f"  - {d.name}")
        sys.exit(1)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Processing v2 dataset:")
    print(f"  Data path: {data_path}")
    print(f"  Metallicity: {metallicity} (Z = {METALLICITY_VALUES[metallicity]})")
    print(f"  Include metallicity column: {include_metallicity}")
    print(f"  Output file: {output_file}")

    # Find all .params files
    params_files = sorted(metallicity_dir.glob("*.params"))
    model_ids = [p.stem for p in params_files]
    n_models = len(model_ids)

    print(f"  Found {n_models} models in {metallicity_dir.name}/")

    if n_models == 0:
        print(f"ERROR: No .params files found in {metallicity_dir}")
        sys.exit(1)

    params = [None] * n_models
    metallicity_value = METALLICITY_VALUES[metallicity]

    with h5py.File(output_file, "w") as fh:
        print("\nProcessing models...")

        for idx, model_id in enumerate(tqdm(model_ids, desc="Converting models")):
            # Read parameters
            params[idx] = np.genfromtxt(metallicity_dir / f"{model_id}.params")

            # Read data files
            for datatype in ["pdr", "spop"]:
                fin_file = metallicity_dir / f"{model_id}.{datatype}.fin"

                if fin_file.exists():
                    pdr_data = np.genfromtxt(fin_file)

                    # Prepend metallicity column if requested
                    if include_metallicity:
                        n_rows = pdr_data.shape[0]
                        metallicity_col = np.full(
                            (n_rows, 1), metallicity_value, dtype=np.float32
                        )
                        pdr_data = np.hstack([metallicity_col, pdr_data])

                    fh.create_dataset(
                        name=f"{model_id}/{datatype}", data=pdr_data, dtype="float32"
                    )
                else:
                    print(f"\nWarning: {fin_file} not found")

        # Store metadata
        fh.create_dataset("model_df", data=np.array(params), dtype="float32")
        fh.create_dataset("model_ids", data=model_ids, dtype="S10")
        fh.create_dataset("species", data=SPECIES, dtype="S10")

        # Store metallicity info
        fh.attrs["metallicity"] = metallicity
        fh.attrs["metallicity_value"] = metallicity_value
        fh.attrs["includes_metallicity_column"] = include_metallicity

    print(f"\n✓ Successfully created {output_file}")
    print(f"  - Metallicity: {metallicity} (Z = {metallicity_value})")
    print(f"  - Models: {n_models}")
    print(f"  - Species: {len(SPECIES)}")
    print(f"  - Metallicity column: {'Yes' if include_metallicity else 'No'}")


def main():
    parser = argparse.ArgumentParser(
        description="Process v2 dataset with metallicity selection"
    )
    parser.add_argument(
        "data_path", type=str, help="Path to directory with metallicity subdirectories"
    )
    parser.add_argument("output_file", type=str, help="Path to output HDF5 file")
    parser.add_argument(
        "--metallicity",
        type=str,
        default="Z1p0",
        choices=list(METALLICITY_VALUES.keys()),
        help="Metallicity to process (default: Z1p0 = solar)",
    )
    parser.add_argument(
        "--include-metallicity",
        action="store_true",
        help="Prepend metallicity value as first column in time series data",
    )
    parser.add_argument(
        "--all-metallicities",
        action="store_true",
        help="Process ALL metallicities with appended dataset names (overrides --metallicity)",
    )

    args = parser.parse_args()

    if args.all_metallicities:
        process_all_metallicities(
            args.data_path, args.output_file, args.include_metallicity
        )
    else:
        process_v2_dataset(
            args.data_path,
            args.output_file,
            args.metallicity,
            args.include_metallicity,
        )


if __name__ == "__main__":
    main()
