import io
import sys
from pathlib import Path

import h5py
import numpy as np

BUFFER_SIZE = 64

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
    data_path = Path(sys.argv[1])
    store_path = Path(sys.argv[2])

    line_index = 0
    magic_line_length = 0
    buffer_index = 0
    break_condition = False
    pdr_buffer = {}

    with open(data_path, "r") as f:
        while True:
            header_line = f.readline()
            if not header_line:
                break_condition = True
            if not break_condition:
                if magic_line_length:
                    _, _, model_id, number_of_lines = header_line.split()
                    model_name = f"model{int(model_id):07d}"
                    text = f.read(magic_line_length * (int(number_of_lines) + 1))
                    assert all(
                        [len(t) == magic_line_length - 1 for t in text.split("\n")[:-1]]
                    )
                    pdr_buffer[model_name] = np.genfromtxt(io.StringIO(text))
                else:
                    # Get the line length of the first file
                    magic_line_length = len(f.readline())
                    print(f"The magic line length is {magic_line_length}")
                    # Reset the seek position to 0, effectively restarting the process, but now with the read buffer set correctly.
                    f.seek(0)
            if len(pdr_buffer) >= BUFFER_SIZE or break_condition:
                # Write everything every buffer_size or after break condition is met:
                print(f"writing models {list(pdr_buffer.keys())}")
                for model_name, pdr_data in pdr_buffer.items():
                    with h5py.File(store_path, "a") as fh:
                        fh.create_dataset(
                            name=f"{model_name}/pdr",
                            data=pdr_data,
                            dtype="float32",
                            compression="gzip",
                        )
                pdr_buffer = {}
            if break_condition:
                break
