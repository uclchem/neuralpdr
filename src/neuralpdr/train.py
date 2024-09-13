# Standard imports
import argparse
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import yaml
from callbacks import EarlyTerminate, NeptuneLogger, OneBatchPlotter, SaveWeightCallback
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

# Potential debugging mode
# logging.basicConfig(level=logging.DEBUG)


@eqx.filter_value_and_grad
def grad_loss(model: eqx.Module, avs: jax.Array, batch_data: jax.Array) -> jax.Array:
    """Compute the loss function for the NeuralODE.

    Args:
        model (eqx.Module): The NN part of the neuralODE
        avs (jax.Array): One batch of visual extinctions
        batch_data (jax.Array): One batch of timeseries data

    Returns:
        jax.Array: MSE for this batch
    """
    pred_batch_data = eqx.filter_vmap(solve_ODE, in_axes=(None, 0, 0), out_axes=0)(
        model, avs, batch_data[:, 0, :]
    )
    return jnp.mean((pred_batch_data - batch_data) ** 2)


@eqx.filter_jit
def grad_loss_only(
    model: eqx.Module, avs: jax.Array, batch_data: jax.Array
) -> jax.Array:
    """Compute the loss function for the NeuralODE.

    Args:
        model (eqx.Module): The NN part of the neuralODE
        avs (jax.Array): One batch of visual extinctions
        batch_data (jax.Array): One batch of timeseries data

    Returns:
        jax.Array: MSE for this batch
    """
    pred_batch_data = eqx.filter_vmap(solve_ODE, in_axes=(None, 0, 0), out_axes=0)(
        model, avs, batch_data[:, 0, :]
    )
    return jnp.mean((pred_batch_data - batch_data) ** 2)


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    avs: jax.Array,
    batch_data: jax.Array,
) -> tuple[jax.Array, eqx.Module, optax.OptState]:
    """Make a step in the optimization process.

    Args:
        model (eqx.Module): The NN part of the neuralODE
        optim (optax.GradientTransformation): The optimizer
        opt_state (optax.OptState): The state of the optimizer
        avs (jax.Array): One batch of visual extinctions
        batch_data (jax.Array): One batch of timeseries data

    Returns:
        tuple[jax.Array, eqx.Module, optax.OptState]: The loss, the updated model and the updated optimizer state
    """
    value, grads = grad_loss(model, avs, batch_data)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return value, model, opt_state


# @eqx.filter_jit # This needs to be converted to a scan if we want to jit it effectively.
def do_epoch(
    epoch: int,
    mlp: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    av_train_batches: list[jnp.array],
    data_train_batches: list[jnp.array],
    av_val_batches: list[jnp.array],
    data_val_batches: list[jnp.array],
) -> tuple[float, float, eqx.Module, optax.OptState]:
    """Perform an epoch of training on the NeuralODE.

    Args:
        epoch (int): The current epoch
        mlp (eqx.Module):  The neural network
        optim (optax.GradientTransformation): The optimizer
        opt_state (optax.OptState): The state of the optimizer
        av_train_batches (list[jnp.array]): Visual extinction batches for training
        data_train_batches (list[jnp.array]): Data batches for training
        av_val_batches (list[jnp.array]): Visual extinction batches for validation
        data_val_batches (list[jnp.array]): Data batches for validation

    Returns:
        tuple[float, float, eqx.Module, optax.OptState]: The training loss, validation loss, updated neural network and optimizer state
    """
    train_losses = jnp.zeros((len(av_train_batches),))
    for step, (av, data) in enumerate(zip(av_train_batches, data_train_batches)):
        train_value, mlp, opt_state = make_step(mlp, optim, opt_state, av, data)
        train_losses = train_losses.at[step].set(train_value)
        # train_losses[step] = train_value # for numpy arrays instead.
    train_loss = jnp.mean(train_losses)
    val_loss = eqx.filter_vmap(grad_loss_only, in_axes=(None, 0, 0))(
        mlp, av_val_batches, data_val_batches
    )
    return train_loss, jnp.mean(val_loss), mlp, opt_state


# Function to train the NeuralODE
def train(
    mlp: eqx.Module,
    opt_state: optax.OptState,
    epochs: list[int],
    fracs: list[float],
    train_loader: PDRLoader,
    val_loader: PDRLoader,
    shuffle_every_n_epochs: int = None,
    save_file_path: Path = None,
    optim: optax.GradientTransformation = None,
    on_epoch_end_callback: Callable = None,
    early_termination_callback: Callable = None,
):
    """Train the NeuralODE

    Args:
        mlp (eqx.Module): The neural network.
        opt_state (optax.OptState): The state of the optimizer.
        epoch_checkpoints (list[int]): List of epoch checkpoints.
        fracs (list[float]): List of fractions for visual extinctions.
        train_loader (PDRLoader): The data loader for training data.
        val_loader (PDRLoader): The data loader for validation data.
        shuffle_every_n_epochs (int, optional): Number of epochs after which to shuffle the training data. Defaults to None.
        loss_type (str, optional): Type of loss function to use. Defaults to None.
        visualize (bool, optional): Whether to visualize the training progress. Defaults to True.
        save_file_path (Path, optional): Path to save the training progress. Defaults to None.
        optim (optax.GradientTransformation, optional): The optimizer to use. Defaults to None.
        end_of_epoch_callback (Callable, optional): Callback function to execute at the end of each epoch. Defaults to None.
    """
    # For training on specific chunks of the dataset to avoid getting caught in local minima
    epoch_checkpoints_a = [0] + list(np.cumsum(epochs, dtype=int)[:-1])
    epoch_checkpoints_b = list(np.cumsum(epochs, dtype=int))

    for frac, epoch_a, epoch_b in zip(fracs, epoch_checkpoints_a, epoch_checkpoints_b):
        train_loader.set_timeseries_fraction(frac)
        val_loader.set_timeseries_fraction(frac)
        for epoch in range(epoch_a, epoch_b + 1):
            if (
                epoch > 0
                and shuffle_every_n_epochs
                and epoch % shuffle_every_n_epochs == 0
            ):
                train_loader.shuffle_batches()
            av_train_batches, data_train_batches = train_loader.get_all_batches()
            av_val_batches, data_val_batches = val_loader.get_all_batches()
            train_loss, val_loss, mlp, opt_state = do_epoch(
                epoch,
                mlp,
                optim,
                opt_state,
                av_train_batches,
                data_train_batches,
                av_val_batches,
                data_val_batches,
            )
            print(
                f"{datetime.now()} Epoch: {epoch}, train Loss: {train_loss}, val Loss: {val_loss}"
            )
            if on_epoch_end_callback:
                for cb in on_epoch_end_callback:
                    cb(epoch, mlp, train_loss, val_loss, train_loader, val_loader)
            if early_termination_callback:
                early_termination_callback(
                    epoch, mlp, train_loss, val_loss, train_loader, val_loader
                )
                if early_termination_callback.get_stopcondition():
                    return train_loss, val_loss

    return train_loss, val_loss


def make_predictions(mlp, av, data):
    pred = eqx.filter_vmap(solve_ODE, in_axes=(None, 0, 0))(mlp, av, data[:, 0, :])
    return pred


def get_git_info() -> dict[str, str]:
    """Get the git info of the current repository

    Returns:
        dict[str, str]: Dict with keys 'git_commit_hash', 'git_info' and 'git_diff_to_HEAD' with the corresponding values
    """
    git_info = {}
    try:
        git_info["git_commit_hash"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
        git_info["git_info"] = (
            subprocess.check_output(["git", "log", "-1"]).decode().strip()
        )
        git_info["git_diff_to_HEAD"] = (
            subprocess.check_output(["git", "diff", "HEAD"]).decode().strip()
        )
    except Exception as e:
        print(f"Error getting git info: {e}")
    return git_info


def get_config():
    parser = argparse.ArgumentParser(description="Parse configuration file")
    parser.add_argument(
        "config_file", type=str, help="Path to the configuration yaml file"
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    return config


def main(args=None):
    if not args:
        args = get_config()
    index_range = [args["start_index"], args["end_index"]]
    save_file_path = Path(args["save_file_path"])
    save_file_path.mkdir(parents=True, exist_ok=True)

    batch_size = args["batch_size"]
    depth = args["depth"]
    width = args["width"]
    dataset_path = Path(args["dataset_path"])
    learning_rate = args["learning_rate"]
    weight_scale = args["weight_scale"]
    weight_truncation = args["weight_truncation"]
    shuffle_every_n_epochs = args["shuffle_every_n_epochs"]
    weight_decay = args["weight_decay"]
    epochs_per_partial_timeseries = args["epochs_per_partial_timeseries"]
    learning_rate_scheme = args["learning_rate_scheme"]
    double_epochs_last_fraction = args["double_epochs_last_fraction"]
    timeseries_fractions = args["timeseries_fractions"]

    # Convert two of the parameters from numpy values to python defaults for json:
    args["learning_rate_scheme"] = str(args["learning_rate_scheme"])
    args["double_epochs_last_fraction"] = bool(args["double_epochs_last_fraction"])

    # Some hard-coded bits
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    epochs = [epochs_per_partial_timeseries for _ in timeseries_fractions]
    if double_epochs_last_fraction:
        epochs[-1] *= 2
    args["timeseries_fractions"] = timeseries_fractions
    args["epoch_schedule"] = epochs

    input_features = [
        "visual_extinction",
        "tgas",
        "tdust",
        "density",
        "radfield",
        "zeta_init",
        "CH3OH",
        "CS",
        "CO+",
        "H2CO",
        "HCO",
        "C2",
        "HCN",
        "NH",
        "HCO+",
        "CN",
        "O2",
        "CH",
        "H2O",
        "C",
        "CO",
        "O",
        "H2",
        "H",
        "e-",
    ]
    args["input_features"] = input_features
    # Add a small epsilon to the visual extinction to avoid log errors
    av_eps = 1e-11
    args["av_eps"] = input_features
    # Add a larger epsilon to av in the time series and the rest of the data
    # WARNING: this is hard coded on Av being the 0 index.
    data_eps = [
        1e-10,
    ] + [
        1e-20,
    ] * (len(input_features) - 1)
    args["data_eps"] = data_eps

    # After parameters are set, get the information of the git repository.
    args.update(get_git_info())

    # Load the dataframe with each of the model parameters.
    model_df = h5py_load(
        dataset_path, "model_df", return_dataframe=True, columns=["g_uv", "n_H", "zeta"]
    )
    model_df.index = h5py_load(dataset_path, "model_ids", text=True)
    model_df.columns = ["radfield_init", "density_init", "zeta_init"]

    train_indices, val_indices, test_indices = shuffle_and_split(
        model_df, train_split=train_split, val_split=val_split, test_split=test_split
    )

    # Loading data
    train_dataloader = PDRLoader(
        dataset_path,
        model_df,
        index_range=index_range,
        model_indices=train_indices.index.to_list(),
        input_features=input_features,
        batch_size=batch_size,
        stage="train",
        av_normalization_kwargs=dict(eps=av_eps),
        data_normalization_kwargs=dict(eps=data_eps),
    )
    norm_parameters = train_dataloader.get_normalization()
    val_dataloader = PDRLoader(
        dataset_path,
        model_df,
        index_range=index_range,
        model_indices=val_indices.index.to_list(),
        input_features=input_features,
        batch_size=batch_size,
        stage="val",
        av_normalization_kwargs=dict(eps=av_eps, **norm_parameters["av"]),
        data_normalization_kwargs=dict(eps=data_eps, **norm_parameters["data"]),
    )
    test_dataloader = PDRLoader(
        dataset_path,
        model_df,
        index_range=index_range,
        model_indices=test_indices.index.to_list(),
        input_features=input_features,
        batch_size=batch_size,
        stage="test",
        av_normalization_kwargs=dict(eps=av_eps, **norm_parameters["av"]),
        data_normalization_kwargs=dict(eps=data_eps, **norm_parameters["data"]),
    )
    # Write all metadata of the trainer, so we can reproduce the datasets later.
    with open(save_file_path / "data_metadata.json", "w") as fh:
        fh.write(
            json.dumps(
                dict(
                    train_indices=train_indices.index.to_list(),
                    val_indices=val_indices.index.to_list(),
                    test_indices=test_indices.index.to_list(),
                    norm_parameters={
                        kk: {k: v.tolist() for k, v in vv.items()}
                        for kk, vv in norm_parameters.items()
                    },
                )
            )
        )

    # Defining the MLP and optimizer
    model_key = jrandom.PRNGKey(0)
    mlp = get_model(len(input_features), width, depth, model_key, activation=jax.nn.softplus)

    # add callbacks for various things.
    save_weights_callback = SaveWeightCallback(save_file_path, args, 10)
    neptune_callback = NeptuneLogger("AutoChemulator/3dpdr", args, tags=args.get("neptune_tags"))
    plot_callback = OneBatchPlotter(save_file_path, 10)
    early_terminate_callback = EarlyTerminate(100, patience=10)

    # Weight initialization
    weight_initializer = trunc_init(weight_scale, -weight_truncation, weight_truncation)
    mlp = init_linear_weight(mlp, weight_initializer, model_key)
    # Training the network and saving the loss functions

    # Scheduler:
    if learning_rate_scheme == "constant":
        learning_rate_scheduler = optax.constant_schedule(learning_rate)
    elif learning_rate_scheme == "sgdr":
        learning_rate_scheduler = optax.sgdr_schedule(
            [
                dict(
                    init_value=0.1 * learning_rate,
                    peak_value=learning_rate,
                    exponent=1e-1,
                    warmup_steps=10 * len(train_dataloader),
                    decay_steps=i * len(train_dataloader),
                )
                for i in epochs
            ]
        )
    else:
        raise ValueError("Invalid learning rate scheme")

    # optim = optax.adabelief(learning_rate=cosine_annealing_scheduler)
    optim = optax.adamw(
        learning_rate=learning_rate_scheduler, weight_decay=weight_decay
    )
    opt_state = optim.init(eqx.filter(mlp, eqx.is_array))

    train_loss, val_loss = train(
        mlp,
        opt_state,
        epochs,
        timeseries_fractions,
        train_dataloader,
        val_dataloader,
        save_file_path=save_file_path,
        optim=optim,
        shuffle_every_n_epochs=shuffle_every_n_epochs,
        on_epoch_end_callback=[
            save_weights_callback,
            neptune_callback,
            plot_callback,
        ],
        early_termination_callback=early_terminate_callback,
    )
    return train_loss, val_loss


if __name__ in "__main__":
    main()
