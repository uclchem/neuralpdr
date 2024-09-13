# Standard imports
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jaxlib
import numpy as np
import optax
from callbacks import NeptuneLogger, OneBatchPlotter, SaveWeightCallback
from model import get_model, init_linear_weight, solve_ODE, trunc_init

from data import PDRLoader, h5py_load, shuffle_and_split

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

def make_predictions(mlp, av, data):
    pred = eqx.filter_vmap(solve_ODE, in_axes=(None, 0, 0))(mlp, av, data[:, 0, :])
    return pred


def deserializer(hyperparameters_path, weights_path):
    with open(hyperparameters_path, "rb") as fh:
        hp = json.loads(fh.read())

    dummy_model = get_model(
        n_input_features=len(hp["input_features"]),
        depth=hp["depth"],
        width=hp["width"],
        model_key=jrandom.PRNGKey(0),
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
    parser.add_argument(
        "--weights_file",
        type=str,
        help="Weights file to load"
    )
    return parser.parse_args()


def main(args=None):
    dataset_path = Path(args["dataset_path"])
    data_configuration = Path(args["model_dir"]) / "data_metadata.json"
    model_configuration = Path(args["model_dir"]) / "hyperparameters.json"
    model_weights_path = Path(args["model_dir"]) / args["weights_file"]
    
    mlp, hyperparameters = deserializer(model_configuration, model_weights_path)

    input_features = hyperparameters["input_features"] 
    
    # Add a small epsilon to the visual extinction to avoid log errors
    av_eps = 1e-11
    # Add a larger epsilon to av in the time series and the rest of the data
    # WARNING: this is hard coded on Av being the 0 index.
    data_eps = [
        1e-10,
    ] + [
        1e-20,
    ] * (len(input_features) - 1)

    model_df = h5py_load(
        dataset_path, "model_df", return_dataframe=True, columns=["g_uv", "n_H", "zeta"]
    )
    model_df.index = h5py_load(dataset_path, "model_ids", text=True)
    model_df.columns = ["radfield_init", "density_init", "zeta_init"]

    with open(data_configuration, "r") as fh:
        data_metadata = json.loads(fh.read())
    train_keys = data_metadata["train_indices"]
    val_keys = data_metadata["val_indices"]
    test_keys = data_metadata["test_indices"]
    normalisation = data_metadata["norm_parameters"]
    normalisation = {kk: {k:np.array(v) for k, v in vv.items()} for kk, vv in normalisation.items()}

    test_dataloader = PDRLoader(
        dataset_path,
        model_df,
        index_range=[0, 300],
        model_indices=test_keys,
        input_features=input_features,
        batch_size=128,
        stage="test",
        av_normalization_kwargs=dict(eps=av_eps, **normalisation["av"]),
        data_normalization_kwargs=dict(eps=data_eps, **normalisation["data"]),
    )

    # Defining the MLP and optimizer

    #
    avs, datas = test_dataloader.get_all_batches()
    preds = []
    for av, data in zip(avs, datas):
        preds.append(make_predictions(mlp, av, data))
        
    # Unnormalise the predictions
    preds_unnormed = [test_dataloader.inv_normalize_data(pred) for pred in preds]
    datas_unnormed = [test_dataloader.inv_normalize_data(data) for data in datas]
    av_unnormed = [test_dataloader.inv_normalize_av(av) for av in avs]
    

    # Unroll av, data and preds :
    avs = np.vstack(avs)
    datas = np.vstack(datas)
    preds = np.vstack(preds)
    avs_unnormed = np.vstack(av_unnormed)
    datas_unnormed = np.vstack(datas_unnormed)
    preds_unnormed = np.vstack(preds_unnormed)

    # Save the predictions
    print(avs.shape, datas.shape, preds.shape)
    print(avs_unnormed.shape, datas_unnormed.shape, preds_unnormed.shape)
    savepath = model_weights_path.parent / f"{model_weights_path.stem}_predictions.h5"
    print(savepath)
    with h5py.File(savepath, "w") as fh:
        fh.create_dataset("avs_normalized", shape=avs.shape, data=avs, dtype=np.float32)
        fh.create_dataset("datas_normalized", shape=datas.shape, data=datas, dtype=np.float32)
        fh.create_dataset("preds_normalized", shape=preds.shape, data=preds, dtype=np.float32)
        fh.create_dataset("avs", shape=avs_unnormed.shape, data=avs_unnormed, dtype=np.float32)
        fh.create_dataset("datas", shape=datas_unnormed.shape, data=datas_unnormed, dtype=np.float32)
        fh.create_dataset("preds", shape=preds_unnormed.shape, data=preds_unnormed, dtype=np.float32)
        
                # Save the normalisation parameters
        fh.create_dataset("metadata", data=np.array([]))
        import yaml
        fh["metadata"].attrs["data_metadata"] = yaml.dump(data_metadata)
        fh["metadata"].attrs["hyperparameters"] = yaml.dump(hyperparameters)


if __name__ in "__main__":
    main(vars(get_parser()))
