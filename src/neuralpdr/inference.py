# Standard imports
import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import equinox as eqx
import h5py
import jax
from jaxtyping import Array, ArrayLike
import numpy as np
from tqdm import tqdm

from neuralpdr.config import (
    AUXNorm,
    Activation,
    DataMetadata,
    DataNorm,
    Features,
    IVNorm,
    Latent,
    Norms,
    read_conf,
    read_as,
    to_json,
)
from neuralpdr.data import (
    PDRLoader,
    log_semi_sorter,
    pad_and_stack,
)
from neuralpdr.model import (
    EncoderEvolveDecoder,
    # get_model,
    # init_linear_weight,
    # solve_ODE,
    # trunc_init,
)

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_while_loop_double_buffering=true "
)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# Jax backend
jax.config.update("jax_platform_name", "gpu")
# Enable double precision for greater numerical stability solving the ODEs
# jax.config.update("jax_enable_x64", True)
# XLA flags for better performance
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_while_loop_double_buffering=true "
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Potential debugging mode
# logging.basicConfig(level=logging.DEBUG)

_ACTIVATION: dict[Activation, Callable[[ArrayLike], Array]] = {
    "tanh": jax.nn.tanh,
    "softplus": jax.nn.softplus,
}


@eqx.filter_jit
def make_predictions(mlp, batch_iv, batch_data, batch_aux):
    pred_y, evolved_z, auto_y, direct_z, steps = eqx.filter_vmap(
        mlp, in_axes=(0, 0, 0), out_axes=0
    )(batch_iv[:, :, 0], batch_data, batch_aux)
    return {
        "pred_y": pred_y,
        "evolved_z": evolved_z,
        "auto_y": auto_y,
        "direct_z": direct_z,
        "steps": steps,
    }


def checkpoint_deserializer(hyperparameters_path, weights_path):
    match read_conf(hyperparameters_path):
        case Latent() as hp:
            ...
        case _c:
            raise RuntimeError(f"{type(_c)}: unsupported config for inference")

    input_features = read_as(hp.input_features_file, Features)
    key = jax.random.PRNGKey(0)
    mlp_key, enc_key, dec_key = jax.random.split(key, 3)
    dummy_model = EncoderEvolveDecoder(
        input_features.data,
        hp.enc_dec_width,
        hp.enc_dec_depth,
        hp.width,
        hp.depth,
        hp.weight_scale,
        hp.weight_truncation,
        keys=[mlp_key, enc_key, dec_key],
        latent_bottleneck=hp.bottleneck,
        n_aux_features=len(input_features.aux),
        latent_final_activation=_ACTIVATION[hp.final_activation],
    )

    with open(weights_path, "rb") as fh:
        model = eqx.tree_deserialise_leaves(fh, dummy_model)
    return model, hp


def get_parser():
    # Defining the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory with the outputs",
    )
    parser.add_argument("--weights_file", type=str, help="Weights file to load")
    return parser.parse_args()


def main(opts: argparse.Namespace):
    dataset_path = Path(opts.dataset_path)
    data_configuration = Path(opts.model_dir) / "data_metadata.json"
    model_configuration = Path(opts.model_dir) / "hyperparameters.json"
    if Path(opts.weights_file).exists():
        model_weights_path = opts.weights_file
    else:
        model_weights_path = Path(opts.model_dir) / opts.weights_file

    mlp, hyperparameters = checkpoint_deserializer(
        model_configuration, model_weights_path
    )

    input_features = read_as(hyperparameters.input_features_file, Features)

    if hyperparameters.normalisations_file:
        normalization_parameters = read_as(hyperparameters.normalisations_file, Norms)
    else:
        normalization_parameters = Norms(IVNorm(), AUXNorm(), DataNorm())

    # model_indices = text_from_h5(dataset_path, "model_ids")
    # model_df.columns = ["zeta_init", "radfield_init", "density_init"]

    # # Only select models that are longer than 32 timesteps.
    # model_indices = filter_models_by_series_length(
    #     hyperparameters["minimal_timeseries_length"], model_indices, dataset_path
    # )

    # Add a small epsilon to the visual extinction to avoid log errors
    data_metadata = read_as(data_configuration, DataMetadata)
    train_keys = data_metadata.train_indices
    val_keys = data_metadata.val_indices
    test_keys = data_metadata.test_indices

    # In order to not eat all memory with preds, we load everything in batches:

    keys = train_keys + val_keys + test_keys

    keys_splits = np.array_split(keys, 25)
    savepath = model_weights_path.parent / f"{model_weights_path.stem}_predictions3.h5"
    inference_time = timedelta(0.0)
    with h5py.File(dataset_path, "r") as fh_data:
        data_header = [s.decode("utf8") for s in fh_data["header"]]
        prediction_original_indices = {
            feature: data_header.index(feature) for feature in input_features.data
        }
        with h5py.File(savepath, "w") as fh_save:
            for key_split in keys_splits:
                dataloader = PDRLoader(
                    dataset_path=dataset_path,
                    independent_variable=input_features.iv,
                    data_features=input_features.data,
                    auxiliary_features=input_features.aux,
                    index_range=(
                        hyperparameters.start_index,
                        hyperparameters.end_index,
                    ),
                    model_indices=list(key_split),  # test_keys[:1024],
                    # model_indices=hyperparameters["val_indices"],
                    batch_size=128,
                    stage="val",
                    independent_variable_normalization_kwargs=asdict(
                        normalization_parameters.iv
                    ),
                    features_normalization_kwargs=asdict(normalization_parameters.data),
                    auxiliary_features_normalization_kwargs=asdict(
                        normalization_parameters.aux
                    ),
                    collate_fn=pad_and_stack,
                    batch_permutation_function=log_semi_sorter,  # lambda x, y: x,  # do not shuffle
                    use_cache=False,
                    drop_last=False,
                )

                ivs, datas, auxs = dataloader.get_all_batches()
                keys_per_batch = dataloader.get_batch_keys()
                # outputs_per_key = {
                #     key: {}
                #     for key in ["pred_y", "evolved_z", "auto_y", "direct_z", "steps"]
                # }
                outputs_per_model: dict[str, dict] = {}
                for ivs, data, aux, batch_keys in tqdm(
                    zip(ivs, datas, auxs, keys_per_batch)
                ):
                    t1 = datetime.now()
                    outputs = make_predictions(mlp, ivs, data, aux)
                    t2 = datetime.now()
                    inference_time += t2 - t1
                    # Split each batch into individual arrays:
                    for key in batch_keys:
                        outputs_per_model[str(key)] = {}
                    for output_key in outputs:
                        split_output = np.split(
                            outputs[output_key], outputs[output_key].shape[0]
                        )
                        for key, output in zip(batch_keys, split_output):
                            outputs_per_model[str(key)][output_key] = output

                    print(f"Batch took {t2 - t1}")
                t3 = datetime.now()
                counter = 0
                for model_key, model_data in outputs_per_model.items():
                    # Original dataset:
                    original_pdr: np.ndarray = fh_data[model_key + "/pdr"][:]
                    fh_save.create_dataset(
                        str(model_key) + "/pdr",
                        data=original_pdr,
                        dtype=np.float32,
                    )

                    pred_y = dataloader.inv_normalize(model_data["pred_y"], "data")

                    if len(original_pdr) > pred_y.shape[1]:
                        print("warning: ", model_key, pred_y.shape, original_pdr.shape)
                    end_index = min(len(original_pdr), model_data["pred_y"].shape[1])
                    # print(model_key, pred_y.shape, original_pdr.shape, counter)
                    counter += 1
                    for src_index, (header, target_index) in enumerate(
                        prediction_original_indices.items()
                    ):
                        original_pdr[:end_index, target_index] = pred_y[
                            0, :end_index, src_index
                        ]
                    fh_save.create_dataset(
                        str(model_key) + "/pdr_pred",
                        data=original_pdr,
                        dtype=np.float32,
                    )
                    auto_y = dataloader.inv_normalize(model_data["auto_y"], "data")
                    for src_index, (header, target_index) in enumerate(
                        prediction_original_indices.items()
                    ):
                        original_pdr[:end_index, target_index] = auto_y[
                            0, :end_index, src_index
                        ]
                    fh_save.create_dataset(
                        str(model_key) + "/pdr_auto",
                        data=original_pdr,
                        dtype=np.float32,
                    )
                    fh_save.create_dataset(
                        str(model_key) + "/evolved_z",
                        data=model_data["evolved_z"][0, :end_index],
                        dtype=np.float32,
                    )
                    fh_save.create_dataset(
                        str(model_key) + "/direct_z",
                        data=model_data["direct_z"][0, :end_index],
                        dtype=np.float32,
                    )
                print("Writing this split took: ", datetime.now() - t3)
            # Save the normalisation parameters
            fh_save.create_dataset("metadata", data=np.array([]))

            fh_save["metadata"].attrs["data_metadata"] = json.dumps(
                asdict(data_metadata)
            )
            fh_save["metadata"].attrs["hyperparameters"] = to_json(
                hyperparameters
            ).decode()

    print(f"Pure inference time was: {inference_time}")


if __name__ in "__main__":
    parser = get_parser()
    main(parser)
