from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import make_smoothing_spline
from tqdm import tqdm


def make_log_smooth_spline(x, y, lam, lower_bound=-32.0, higher_bound=32.0):
    spl = make_smoothing_spline(x, np.log10(y + np.power(10.0, lower_bound)), lam=lam)
    data = spl(x)
    data[data < lower_bound] = lower_bound
    data[data > higher_bound] = higher_bound
    return 10**data


def smooth_data(data, iv_index, data_indices, aux_indices):
    increase_only_mask = np.diff(data[:, iv_index], prepend=1) > 0.0
    if not (increase_only_mask).all():
        data = data[increase_only_mask]
    for idx in data_indices:
        data[:, idx] = make_log_smooth_spline(
            data[:, iv_index], data[:, idx], 1e-4, -30.0, 0.0
        )
    for idx in aux_indices:
        data[:, idx] = make_log_smooth_spline(data[:, iv_index], data[:, idx], 1e-4)
    return data


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("file_path", type=str)
    args = arg_parser.parse_args()

    file_path = Path(args.file_path)

    with h5py.File(file_path, "r") as fh:
        HEADER = [s.decode("utf-8") for s in fh["header"][:]]
        SPECIES = [s.decode("utf-8") for s in fh["species"][:]]
        model_keys = [s.decode("utf-8") for s in fh["model_ids"][:]]
    physics_to_smooth = ["tgas", "tdust", "density", "radfield"]
    batch_size = 8192
    batches = np.array_split(model_keys, len(model_keys) // batch_size)
    species_indices = [HEADER.index(spec) for spec in SPECIES]
    physics_indices = [HEADER.index(phys) for phys in physics_to_smooth]
    iv_index = HEADER.index("visual_extinction")
    model_template = "{}/pdr"
    with h5py.File(file_path, "r") as source_fh:
        with h5py.File(
            file_path.parent / (f"{file_path.stem}_smooth{file_path.suffix}"), "w"
        ) as target_fh:
            target_fh.create_dataset("header", data=HEADER, dtype="S32")
            target_fh.create_dataset("species", data=SPECIES, dtype="S32")
            target_fh.create_dataset("model_ids", data=model_keys, dtype="S32")
            for batch in tqdm(batches):
                data_buffer = [None] * len(batch)
                # Load all data)
                for batch_idx, model_idx in enumerate(batch):
                    model_idx = model_template.format(model_idx)
                    data_buffer[batch_idx] = source_fh[model_idx][:]
                new_data_buffer = Parallel(n_jobs=8)(
                    delayed(smooth_data)(
                        data, iv_index, species_indices, physics_indices
                    )
                    for data in data_buffer
                )
                # Save all data
                for model_idx, data in zip(batch, new_data_buffer):
                    model_idx = model_template.format(model_idx)
                    target_fh.create_dataset(model_idx, data=data, dtype="float32")
