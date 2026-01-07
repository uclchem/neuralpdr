#!/usr/bin/env python
"""
Add headers to v2 dataset HDF5 file.

Usage:
    python add_file_headers_v2.py <input_file> <output_file>

Example:
    python add_file_headers_v2.py data/processed/3dpdr_dataset_v2_raw.h5 data/processed/3dpdr_dataset_v2.h5
"""

import sys
from pathlib import Path

import h5py

PHYSICS_HEADER = [
    "visual_extinction",
    "tgas",
    "tdust",
    "density",
    "radfield",
    "zeta_init",
]

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


def add_v2_headers(input_file: str, output_file: str):
    """
    Add headers to v2 dataset.

    Args:
        input_file: Path to input HDF5 file (raw)
        output_file: Path to output HDF5 file (with headers)
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")

    with h5py.File(input_file, "r") as f_in:
        with h5py.File(output_file, "w") as f_out:
            # Add header
            header = PHYSICS_HEADER + SPECIES
            f_out.create_dataset("header", data=header, dtype="S32")

            # Copy all other datasets
            for key in f_in.keys():
                print(f"  Copying: {key}", end="\r")
                f_in.copy(key, f_out)

            print(f"\n✓ Created {output_file}")
            print(f"  - Header: {len(header)} fields")
            print(
                f"  - Models: {len([k for k in f_in.keys() if '/' not in k and k not in ['model_df', 'model_ids', 'species']])}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_file_headers_v2.py <input_file> <output_file>")
        print("\nExample:")
        print(
            "  python add_file_headers_v2.py data/processed/3dpdr_dataset_v2_raw.h5 data/processed/3dpdr_dataset_v2.h5"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    add_v2_headers(input_file, output_file)
