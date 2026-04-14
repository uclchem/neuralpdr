# Standard imports
import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from neuralpdr.callbacks import (
    EarlyTerminate,
    NeptuneLogger,
    OneBatchPlotter,
    SaveWeightCallback,
)
from neuralpdr.config import (
    AUXNorm,
    DataNorm,
    Features,
    FNO,
    IVNorm,
    Latent,
    Norms,
    read_as,
    read_conf,
)
from neuralpdr.data import (
    PDRLoader,
    filter_models_by_series_length,
    text_from_h5,
    log_semi_sorter,
    pad_and_stack,
    shuffle_and_split,
)
from neuralpdr.inference import checkpoint_deserializer
from neuralpdr.model import EncoderEvolveDecoder
from neuralpdr.utils import get_git_info, join_schedules

jax.config.update("jax_traceback_in_locations_limit", -1)

# Jax backend
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_compilation_cache_dir", "./tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.log_compiles(True)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
# Enable double precision for greater numerical stability solving the ODEs
# jax.config.update("jax_enable_x64", True)
# XLA flags for better performance
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_while_loop_double_buffering=true "
)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

# Prepare the sharding:
num_devices = len(jax.local_devices())
devices = mesh_utils.create_device_mesh((num_devices,))

# Data will be split along the batch axis
MESH = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
SPEC = PartitionSpec("batch")
sharding = NamedSharding(MESH, SPEC)  # naming axes of the sharded partition
# replicated = NamedSharding(mesh, P())


# Potential debugging mode
# logging.basicConfig(level=logging.DEBUG)


@eqx.filter_value_and_grad(has_aux=True)
def grad_loss(
    model: eqx.Module,
    batch_iv: jax.Array,
    batch_data: jax.Array,
    batch_aux: jax.Array,
    weight_per_loss: jax.Array = jnp.array([1.0, 1.0, 1.0]),
) -> jax.Array:
    """Compute the loss function for the NeuralODE.

    Args:
        model (eqx.Module): The NN part of the neuralODE
        batch_iv (jax.Array): One batch of independent variable
        batch_data (jax.Array): One batch of feature data
        aux_data (jax.Array): One batch of auxiliary data

    Returns:
        jax.Array: MSE for this batch
    """

    @partial(shard_map, mesh=MESH, in_specs=SPEC, out_specs=SPEC, check_rep=False)
    def loss_sharded(
        batch_iv: jax.Array,
        batch_data: jax.Array,
        batch_aux: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        pred_y, evolved_z, auto_y, direct_z, steps = eqx.filter_vmap(
            model, in_axes=(0, 0, 0), out_axes=0
        )(batch_iv[:, :, 0], batch_data, batch_aux)
        # Catch the zero padded cells:
        valid_time_mask = jnp.diff(batch_iv, axis=1, prepend=1.0) != 0.0
        # Catch the infinite values caused by max_step:
        valid_pred_mask = jnp.isfinite(pred_y[:, :, :1])
        # Obtain the loss for every sample in the batch after timestep 0.
        rollout_loss = jnp.mean(
            valid_time_mask * valid_pred_mask * (pred_y - batch_data) ** 2
        )
        latent_loss = jnp.mean(valid_time_mask * (direct_z - evolved_z) ** 2)
        auto_loss = jnp.mean(valid_time_mask * (auto_y - batch_data) ** 2)
        total_loss = (
            weight_per_loss[0] * rollout_loss
            + weight_per_loss[1] * latent_loss
            + weight_per_loss[2] * auto_loss
        )
        return jnp.stack(
            (
                jax.lax.pmean(total_loss, axis_name="batch"),
                jax.lax.psum(jnp.sum(~valid_pred_mask), axis_name="batch"),
                jax.lax.pmin(jnp.min(steps), axis_name="batch"),
                jax.lax.pmean(jnp.median(steps), axis_name="batch"),
                jax.lax.pmax(jnp.max(steps), axis_name="batch"),
                jax.lax.pmean(rollout_loss, axis_name="batch"),
                jax.lax.pmean(latent_loss, axis_name="batch"),
                jax.lax.pmean(auto_loss, axis_name="batch"),
            )
        )

    out = loss_sharded(batch_iv, batch_data, batch_aux)
    # jax.debug.print("{out}", out=out)
    return out[0], out[1:8]


@eqx.filter_jit
def grad_loss_only(
    model: eqx.Module, batch_iv: jax.Array, batch_data: jax.Array, batch_aux: jax.Array
) -> jax.Array:
    """Compute the loss function for the NeuralODE.

    Args:
        model (eqx.Module): The NN part of the neuralODE
        batch_iv (jax.Array): One batch of independent variable
        batch_data (jax.Array): One batch of feature data
        aux_data (jax.Array): One batch of auxiliary data

    Returns:
        jax.Array: MSE for this batch
    """
    pred_batch_data, _, _, _, _ = shard_map(
        eqx.filter_vmap(model, in_axes=(0, 0, 0), out_axes=0),
        mesh=MESH,
        in_specs=SPEC,
        out_specs=SPEC,
        check_rep=False,
    )(batch_iv[:, :, 0], batch_data, batch_aux)
    valid_mask = jnp.diff(batch_iv, axis=1, prepend=1.0) != 0.0
    return jnp.mean(valid_mask * (pred_batch_data - batch_data[:, :, :]) ** 2)


@eqx.filter_jit(donate="all")
def make_step(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch_iv: jax.Array,
    batch_data: jax.Array,
    batch_aux: jax.Array,
    loss_weights: jax.Array = jnp.array([1.0, 1.0, 1.0]),
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
    (
        (
            value,
            (
                nan_count,
                solver_steps_min,
                solver_steps_med,
                solver_steps_max,
                rollout_loss,
                latent_loss,
                auto_loss,
            ),
        ),
        grads,
    ) = grad_loss(model, batch_iv, batch_data, batch_aux, loss_weights)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return (
        value,
        model,
        opt_state,
        nan_count,
        solver_steps_min,
        solver_steps_med,
        solver_steps_max,
        rollout_loss,
        latent_loss,
        auto_loss,
    )


# @eqx.filter_jit # This needs to be converted to a scan if we want to jit it effectively.
def do_epoch(
    epoch: int,
    mlp: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    train_loader,
    val_loader,
    callbacks: dict[str, list[Callable]] = None,
    multi_objective_scheduler: Callable = None,
    sharding: jax.sharding.Sharding = None,
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
        on_batch_end_callback (Callable, optional): Callback function to execute at the end of each batch. Defaults to None.

    Returns:
        tuple[float, float, eqx.Module, optax.OptState]: The training loss, validation loss, updated neural network and optimizer state
    """
    # Bits to get sharding to work:
    train_losses = jnp.zeros((len(train_loader),))
    # print("The weights per loss term are:", weights_per_loss_term)
    for step, (iv, data, aux) in enumerate(train_loader):
        # Callbacks at the start of the batch
        if "batch_start" in callbacks:
            for cb in callbacks["batch_start"]:
                cb(epoch=epoch, step=step, mlp=mlp)
        # Timing
        start = datetime.now()
        # Initialize the sharding if needed:
        if sharding:
            iv, data, aux = jax.device_put((iv, data, aux), sharding)
        (
            train_value,
            mlp,
            opt_state,
            nan_count,
            solver_steps_min,
            solver_steps_med,
            solver_steps_max,
            rollout_loss,
            latent_loss,
            auto_loss,
        ) = make_step(
            mlp, optim, opt_state, iv, data, aux, multi_objective_scheduler(epoch)
        )
        # Save the loss for this batch
        if not jnp.isnan(train_value):
            train_losses = train_losses.at[step].set(train_value)
        if "batch_end" in callbacks:
            for cb in callbacks["batch_end"]:
                cb(
                    epoch=epoch,
                    step=step,
                    mlp=mlp,
                    train_loss=train_value,
                    neptune_metrics={
                        "nan_count": nan_count,
                        "step": step,
                        "epoch": epoch,
                        "step_train_loss": train_value,
                        "train_time": (datetime.now() - start).total_seconds(),
                        "train_time_per_sample": (
                            datetime.now() - start
                        ).total_seconds()
                        / data.shape[0],
                        "solver_steps_min": solver_steps_min,
                        "solver_steps_med": solver_steps_med,
                        "solver_steps_max": solver_steps_max,
                        "batch_size": data.shape[0],
                        "batch_series_length": data.shape[1],
                        "rollout_loss": rollout_loss,
                        "latent_loss": latent_loss,
                        "auto_loss": auto_loss,
                    },
                )

    train_loss = jnp.mean(train_losses)
    val_losses = jnp.zeros((len(val_loader),))
    for idx, (iv, data, aux) in enumerate(val_loader):
        if sharding:
            iv, data, aux = jax.device_put((iv, data, aux), sharding)
        val_losses = val_losses.at[idx].set(grad_loss_only(mlp, iv, data, aux))
    return train_loss, jnp.mean(val_losses), mlp, opt_state


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
    multi_objective_loss_scheduler: Callable = None,
    callbacks={},
    sharding: jax.sharding.Sharding = None,
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
    epoch_checkpoints_a = [1] + list(np.cumsum(epochs, dtype=int)[:-1] + 1)
    epoch_checkpoints_b = list(np.cumsum(epochs, dtype=int))

    for frac, epoch_a, epoch_b in zip(fracs, epoch_checkpoints_a, epoch_checkpoints_b):
        train_loader.set_timeseries_fraction(frac)
        val_loader.set_timeseries_fraction(frac)
        for epoch in range(epoch_a, epoch_b + 1):
            train_loader.shuffle_batches()
            train_loss, val_loss, mlp, opt_state = do_epoch(
                epoch=epoch,
                mlp=mlp,
                optim=optim,
                opt_state=opt_state,
                multi_objective_scheduler=multi_objective_loss_scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                callbacks=callbacks,
                sharding=sharding,
            )
            print(
                f"{datetime.now()} Epoch: {epoch}, train Loss: {train_loss}, val Loss: {val_loss}"
            )
            learning_rate = optax.tree_utils.tree_get(
                opt_state,
                "learning_rate",
                filtering=lambda path, value: isinstance(value, jnp.ndarray),
            )
            terminate_early = 0
            if "epoch_end" in callbacks:
                for cb in callbacks["epoch_end"]:
                    # If the return code is anything other than 0 / None, we terminate the training.
                    if cb(
                        model=mlp,
                        epoch=epoch,
                        mlp=mlp,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        neptune_metrics={
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "learning_rate": learning_rate,
                        },
                    ):
                        terminate_early += 1
            if terminate_early:
                return train_loss, val_loss
    return train_loss, val_loss


def make_predictions(mlp, iv, data, aux):
    pred = eqx.filter_vmap(mlp, in_axes=(None, 0, 0))(iv, data[:, 0, :], aux)
    return pred


def get_config() -> Latent | FNO:
    parser = argparse.ArgumentParser(description="Parse configuration file")
    parser.add_argument(
        "config_file", type=str, help="Path to the configuration yaml file"
    )
    opts = parser.parse_args()
    return read_conf(opts.config_file)


def main(conf: Latent | FNO):
    index_range = (conf.start_index, conf.end_index)
    save_file_path = conf.save_file_path
    save_file_path.mkdir(parents=True, exist_ok=True)

    dataset_path = str(conf.dataset_path)
    split = conf.train_test_val_split

    input_features = read_as(conf.input_features_file, Features)

    if conf.normalisations_file.exists():
        normalization_parameters = read_as(conf.normalisations_file, Norms)
    else:
        normalization_parameters = Norms(IVNorm(), AUXNorm(), DataNorm())

    # After parameters are set, get the information of the git repository.
    conf.update(get_git_info())

    # Load the dataframe with each of the model parameters.
    model_indices: list[str] = text_from_h5(dataset_path, "model_ids")

    # Only select models that are longer than 32 timesteps.
    model_indices = filter_models_by_series_length(
        conf.minimal_timeseries_length, model_indices, dataset_path
    )

    # TODO: refactor to make this one effective function call instead of three seperate ones.
    train_indices, val_indices, test_indices = shuffle_and_split(
        model_indices=model_indices,
        train_split=split.train,
        val_split=split.validate,
        test_split=split.test,
    )
    # Loading data
    train_dataloader = PDRLoader(
        dataset_path=conf.dataset_path,
        independent_variable=input_features.iv,
        data_features=input_features.data,
        auxiliary_features=input_features.aux,
        index_range=index_range,
        model_indices=train_indices,
        batch_size=conf.batch_size,
        stage="train",
        independent_variable_normalization_kwargs=asdict(normalization_parameters.iv),
        features_normalization_kwargs=asdict(normalization_parameters.data),
        auxiliary_features_normalization_kwargs=asdict(normalization_parameters.aux),
        collate_fn=pad_and_stack,
        batch_permutation_function=log_semi_sorter,
        use_cache=True,
        batch_subsampling=conf.training_batch_subsampling,
    )
    val_dataloader = PDRLoader(
        dataset_path=conf.dataset_path,
        independent_variable=input_features.iv,
        data_features=input_features.data,
        auxiliary_features=input_features.aux,
        index_range=index_range,
        model_indices=val_indices,
        batch_size=conf.batch_size,
        stage="val",
        independent_variable_normalization_kwargs=asdict(normalization_parameters.iv),
        features_normalization_kwargs=asdict(normalization_parameters.data),
        auxiliary_features_normalization_kwargs=asdict(normalization_parameters.aux),
        collate_fn=pad_and_stack,
        batch_permutation_function=log_semi_sorter,
        use_cache=True,
    )
    # Write all metadata of the trainer, so we can reproduce the datasets later.
    with open(save_file_path / "data_metadata.json", "w") as fh:
        fh.write(
            json.dumps(
                dict(
                    train_indices=train_indices,
                    val_indices=val_indices,
                    test_indices=test_indices,
                )
            )
        )

    # add callbacks for various things.
    save_weights_callback = SaveWeightCallback(save_file_path, conf, 1)
    plot_callback = OneBatchPlotter(save_file_path, 1)  # plot_frequency
    early_terminate_callback = EarlyTerminate(100, patience=10)
    neptune_logger = NeptuneLogger(
        conf["neptune_project"], conf, tags=conf.get("neptune_tags")
    )

    callbacks = {
        "batch_start": [],
        "batch_end": [neptune_logger],
        "epoch_end": [
            save_weights_callback,
            plot_callback,
            early_terminate_callback,
            neptune_logger,
        ],
    }

    # Create a schedule to weight the different loss terms.
    # Weights per loss term
    def multi_objective_loss_scheduler(
        epoch: int,
    ) -> jax.Array:
        term0 = (
            [0.04 for i in range(15)]
            + np.linspace(0.04, 1.0, 15).tolist()
            + [1.0 for i in range(70)]
        )
        term1 = [1e-3 for i in range(100)]
        term2 = (
            [1.0 for i in range(15)]
            + np.linspace(1.0, 0.25, 15).tolist()
            + [0.25 for i in range(70)]
        )
        return jnp.array([term0, term1, term2]).T[epoch]

    key = jax.random.PRNGKey(0)
    mlp_key, enc_key, dec_key = jax.random.split(key, 3)
    if conf.checkpoint_file.exists():
        enc_evolve_dec, hp = checkpoint_deserializer(
            Path(conf.save_file_path) / "hyperparameters.json",
            conf.checkpoint_file,
        )
    else:
        match conf:
            case Latent():
                enc_evolve_dec = EncoderEvolveDecoder(
                    input_features.data,
                    conf.enc_dec_width,
                    conf.enc_dec_depth,
                    conf.width,
                    conf.depth,
                    conf.weight_scale,
                    conf.weight_truncation,
                    keys=[mlp_key, enc_key, dec_key],
                    latent_bottleneck=conf.bottleneck,
                    n_aux_features=len(input_features.aux),
                    latent_final_activation=getattr(jax.nn, conf.final_activation),
                )
            case _:
                raise RuntimeError(f"{conf.model}: not yet supported")

    # Scheduler:
    learning_rate_scheduler = []
    boundaries = []
    epochs = []
    timeseries_fractions = []
    for scheme in conf.learning_schemes:
        if scheme.lr_scheduler == "constant":
            learning_rate_scheduler.append(
                optax.constant_schedule(scheme.learning_rate)
            )
            boundaries.append(scheme.epochs * len(train_dataloader))
            epochs.append(scheme.epochs)
            timeseries_fractions.append(scheme.timeseries_fraction)
        elif scheme.lr_scheduler == "sgdr":
            learning_rate_scheduler.append(
                optax.warmup_cosine_decay_schedule(
                    init_value=0.1 * scheme.learning_rate,
                    peak_value=scheme.learning_rate,
                    exponent=1e-1,
                    warmup_steps=scheme.warmup_epochs * len(train_dataloader),
                    decay_steps=scheme.epochs * len(train_dataloader),
                )
            )
            boundaries.append(scheme.epochs * len(train_dataloader))
            epochs.append(scheme.epochs)
            timeseries_fractions.append(scheme.timeseries_fraction)
        else:
            raise ValueError("Invalid learning rate scheme")

    # Only include learning rate schedules past our checkpoints:
    if conf.checkpoint_epoch > 0:
        if any(np.cumsum(epochs) > conf.checkpoint_epoch):
            idx = np.argmax(np.cumsum(epochs) > conf.checkpoint_epoch)
            learning_rate_scheduler = learning_rate_scheduler[idx:]
            boundaries = boundaries[idx:]
            epochs = epochs[idx:]
            timeseries_fractions = timeseries_fractions[idx:]
        print(
            "Since we are continuing an existing schedule, use shortened learning rate schedule:",
            learning_rate_scheduler,
            boundaries,
            epochs,
            timeseries_fractions,
        )
    boundaries = np.cumsum(boundaries)
    learning_rate_scheduler = join_schedules(
        learning_rate_scheduler, boundaries.tolist()
    )
    # optim = optax.adamw(
    #     learning_rate=learning_rate_scheduler, weight_decay=conf.weight_decay
    # )
    optim = optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate_scheduler, weight_decay=conf.weight_decay
    )
    optim = optax.chain(optax.clip_by_global_norm(1.0), optim)
    opt_state = optim.init(eqx.filter(enc_evolve_dec, eqx.is_array))

    train_loss, val_loss = train(
        enc_evolve_dec,
        opt_state,
        epochs,
        timeseries_fractions,
        train_dataloader,
        val_dataloader,
        save_file_path=save_file_path,
        optim=optim,
        multi_objective_loss_scheduler=multi_objective_loss_scheduler,
        shuffle_every_n_epochs=conf.shuffle_every_n_epochs,
        callbacks=callbacks,
        sharding=sharding,
    )
    return train_loss, val_loss


if __name__ in "__main__":
    conf = get_config()
    main(conf)
