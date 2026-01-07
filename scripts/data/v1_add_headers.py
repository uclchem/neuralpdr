#!/usr/bin/env python
"""
Add PHYSICS_HEADER and AUXILARY_HEADER to v1 dataset.

This script reads the v1 HDF5 file (which has species and model_df but no header),
extracts the physics parameters from model_df, and creates a properly structured
header dataset with PHYSICS_HEADER + species + AUXILARY_HEADER.

Usage:
    python add_v1_headers.py <input_file> <output_file>

Example:
    python add_v1_headers.py data/zenodo/v1/3dpdr_dataset_8192.h5 data/processed/3dpdr_dataset_v1.h5
"""

import sys
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Header structure as specified
PHYSICS_HEADER = [
    "time_idx",
    "position",
    "visual_extinction",
    "tgas",
    "tdust",
    "etype",
    "density",
    "radfield",
]

# Auxiliary parameters (these will be added with init values from model_df)
AUXILARY_HEADER = ["radfield_init", "density_init", "zeta_init"]


def add_v1_headers(input_path: str, output_path: str):
    """
    Add header information to v1 dataset.

    Args:
        input_path: Path to input HDF5 file (from data/zenodo/v1/)
        output_path: Path to output HDF5 file (to data/processed/)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading input file: {input_path}")

    with h5py.File(input_path, "r") as f_in:
        # Read species list and model_df
        species = f_in["species"][:]
        model_df = f_in["model_df"][
            :
        ]  # shape: (n_models, 3) - [radfield, density, zeta]
        model_ids = f_in["model_ids"][:]

        n_models = len(model_ids)
        n_species = len(species)

        print(f"Found {n_models} models")
        print(f"Found {n_species} species")
        print(f"Model_df shape: {model_df.shape}")

        # Decode species names
        species_names = [
            s.decode("utf-8") if isinstance(s, bytes) else s for s in species
        ]

        # Construct full header
        full_header = PHYSICS_HEADER + species_names + AUXILARY_HEADER
        n_header = len(full_header)

        print(
            f"Full header length: {n_header} ({len(PHYSICS_HEADER)} physics + {n_species} species + {len(AUXILARY_HEADER)} auxiliary)"
        )

        # Create output file
        print(f"Creating output file: {output_path}")

        with h5py.File(output_path, "w") as f_out:
            # Write header as encoded strings
            header_encoded = np.array(
                [h.encode("utf-8") for h in full_header], dtype="S32"
            )
            f_out.create_dataset("header", data=header_encoded)

            # Write species and model metadata
            f_out.create_dataset("species", data=f_in["species"][:])
            f_out.create_dataset("model_ids", data=model_ids)
            f_out.create_dataset("model_df", data=model_df)

            print("Processing models...")

            # Process each model
            for i, model_id in enumerate(tqdm(model_ids, desc="Copying models")):
                model_id_str = (
                    model_id.decode("utf-8")
                    if isinstance(model_id, bytes)
                    else model_id
                )

                # Copy all data types for this model
                for datatype in ["pdr", "heat", "cool", "line", "opdp", "spop"]:
                    dataset_name = f"{model_id_str}/{datatype}"

                    if dataset_name in f_in:
                        data = f_in[dataset_name][:]
                        f_out.create_dataset(dataset_name, data=data, dtype="float32")
                    else:
                        print(f"Warning: {dataset_name} not found in input file")

                # Add auxiliary data to each model (radfield_init, density_init, zeta_init)
                # These are the initial conditions from model_df
                auxiliary_data = model_df[i]  # [radfield, density, zeta]
                f_out.create_dataset(
                    f"{model_id_str}/auxiliary", data=auxiliary_data, dtype="float32"
                )

    print(f"\n✓ Successfully created {output_path}")
    print(f"  - Header: {n_header} fields")
    print(f"  - Models: {n_models}")
    print("  - Each model now has: pdr, heat, cool, line, opdp, spop, auxiliary")


def main():
    if len(sys.argv) != 3:
        print("Usage: python add_v1_headers.py <input_file> <output_file>")
        print("\nExample:")
        print(
            "  python add_v1_headers.py data/zenodo/v1/3dpdr_dataset_8192.h5 data/processed/3dpdr_dataset_v1.h5"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    add_v1_headers(input_file, output_file)


if __name__ == "__main__":
    main()
