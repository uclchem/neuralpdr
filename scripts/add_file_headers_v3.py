import random
import re
from argparse import ArgumentParser
from pathlib import Path

import h5py

PHYSICS_HEADER = [
    "time_idx",
    "line_idx",
    "x",
    "y",
    "z",
    "position",
    "visual_extinction",
    "tgas",
    "tdust",
    "etype",
    "density",
    "radfield",
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

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("file_path", type=str)
    arg_parser.add_argument("model_match_string", type=str)
    args = arg_parser.parse_args()

    file_path = Path(args.file_path)
    model_match_string = args.model_match_string

    with h5py.File(file_path, "r") as f:
        model_keys = list(f.keys())
        model_keys = [key for key in model_keys if re.match(model_match_string, key)]
        print(
            "A random sample of the {} keys we retrieved:\n".format(len(model_keys)),
            "\n".join(random.choices(model_keys, k=10)),
        )
    with h5py.File(file_path, "a") as fh:
        fh.create_dataset("model_ids", data=model_keys, dtype="S32")
        # fh.create_dataset("model_table", data=model_keys, dtype="float32")
        fh.create_dataset("species", data=SPECIES, dtype="S32")
        fh.create_dataset("header", data=PHYSICS_HEADER + SPECIES, dtype="S32")
