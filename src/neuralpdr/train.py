# Standard imports
import argparse
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Float, Scalar

from neuralpdr.callbacks import (
    EarlyTerminate,
    # NeptuneLogger,
    OneBatchPlotter,
    SaveWeightCallback,
)
from neuralpdr.config import (
    FNO,
    Features,
    Latent,
    Norms,
    read_as,
    read_conf,
    write_as,
)
from neuralpdr.data import (
    PDRLoader,
    SeriesBatch,
    SeriesBatcher,
    SeriesLoader,
    filter_models_by_series_length,
    log_semi_sorter,
    shuffle_and_split,
    text_from_h5,
)
from neuralpdr.inference import checkpoint_deserializer
from neuralpdr.model import EncoderEvolveDecoder, SurrogateModel
from neuralpdr.objective import LatentODEObjective, Objective
from neuralpdr.utils import join_schedules

jax.config.update("jax_traceback_in_locations_limit", -1)

# Jax backend
jax.config.update("jax_platform_name", os.environ.get("JAX_PLATFORM_NAME", "gpu"))
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
    model: SurrogateModel,
    objective: Objective,
    batch: SeriesBatch,
    loss_weights: Float[Array, " W"],
) -> tuple[Scalar, dict[str, Array]]:
    """Compute the training loss of the objective, and its metrics.

    Args:
        model (SurrogateModel): The NN part of the neuralODE
        objective (Objective): The objective that defines the loss and metrics
        batch (SeriesBatch): One batch
        loss_weights (Float[Array, " W"]): The weights of the loss terms

    Returns:
        tuple[Scalar, dict[str, Array]]: The loss and the metrics, reduced over the devices
    """

    @partial(
        shard_map,
        mesh=MESH,
        in_specs=SPEC,
        out_specs=PartitionSpec(),
        check_rep=False,
    )
    def loss_sharded(
        batch: SeriesBatch,
    ) -> tuple[Scalar, dict[str, Array]]:
        loss, metrics = objective(model, batch, loss_weights)
        return jax.lax.pmean(loss, axis_name="batch"), {
            key: objective.metric_reductions[key](value, "batch")
            for key, value in metrics.items()
        }

    return loss_sharded(batch)


@eqx.filter_jit
def grad_loss_only(
    model: SurrogateModel,
    objective: Objective,
    batch: SeriesBatch,
) -> Scalar:
    """Compute the validation loss of the objective.

    Args:
        model (SurrogateModel): The NN part of the neuralODE
        objective (Objective): The objective that defines the validation loss
        batch (SeriesBatch): One batch

    Returns:
        Scalar: The validation loss for this batch
    """
    return shard_map(
        lambda batch: jax.lax.pmean(
            objective.validation_loss(model, batch), axis_name="batch"
        ),
        mesh=MESH,
        in_specs=SPEC,
        out_specs=PartitionSpec(),
        check_rep=False,
    )(batch)


@eqx.filter_jit(donate="all")
def make_step(
    model: SurrogateModel,
    objective: Objective,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch: SeriesBatch,
    loss_weights: Float[Array, " W"],
) -> tuple[Scalar, SurrogateModel, optax.OptState, dict[str, Array]]:
    """Make a step in the optimization process.

    Args:
        model (SurrogateModel): The NN part of the neuralODE
        objective (Objective): The objective that defines the loss
        optim (optax.GradientTransformation): The optimizer
        opt_state (optax.OptState): The state of the optimizer
        batch (SeriesBatch): One batch
        loss_weights (Float[Array, " W"]): The weights of the loss terms

    Returns:
        tuple[Scalar, SurrogateModel, optax.OptState, dict[str, Array]]: The loss, the updated model, the updated optimizer state and the metrics of the objective
    """
    (value, metrics), grads = grad_loss(model, objective, batch, loss_weights)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return value, model, opt_state, metrics


# @eqx.filter_jit # This needs to be converted to a scan if we want to jit it effectively.
def do_epoch(
    epoch: int,
    mlp: SurrogateModel,
    objective: Objective,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    train_loader: SeriesLoader,
    val_loader: SeriesLoader,
    callbacks: dict[str, list[Callable]],
    sharding: jax.sharding.Sharding | None = None,
) -> tuple[Scalar, Scalar, SurrogateModel, optax.OptState]:
    """Perform an epoch of training on the NeuralODE.

    Args:
        epoch (int): The current epoch
        mlp (SurrogateModel):  The neural network
        objective (Objective): The objective that defines the losses and metrics
        optim (optax.GradientTransformation): The optimizer
        opt_state (optax.OptState): The state of the optimizer
        train_loader (SeriesLoader): The data loader for training data
        val_loader (SeriesLoader): The data loader for validation data
        callbacks (dict[str, list[Callable]]): Callback functions to execute at the start ("batch_start") and end ("batch_end") of each batch
        sharding (jax.sharding.Sharding, optional): How to place the batches on the devices. Defaults to None.

    Returns:
        tuple[Scalar, Scalar, SurrogateModel, optax.OptState]: The training loss, validation loss, updated neural network and optimizer state
    """
    # Bits to get sharding to work:
    train_losses = jnp.zeros((len(train_loader),))
    for step, batch in enumerate(train_loader):
        # Callbacks at the start of the batch
        if "batch_start" in callbacks:
            for cb in callbacks["batch_start"]:
                cb(epoch=epoch, step=step, mlp=mlp)
        # Timing
        start = datetime.now()
        # Initialize the sharding if needed:
        if sharding:
            batch = jax.device_put(batch, sharding)
        loss_weights = jnp.asarray(objective.weights(epoch))
        train_value, mlp, opt_state, metrics = make_step(
            mlp, objective, optim, opt_state, batch, loss_weights
        )
        # Save the loss for this batch
        if not jnp.isnan(train_value):
            train_losses = train_losses.at[step].set(train_value)
        if "batch_end" in callbacks:
            data = batch.data
            for cb in callbacks["batch_end"]:
                cb(
                    epoch=epoch,
                    step=step,
                    mlp=mlp,
                    train_loss=train_value,
                    neptune_metrics={
                        "step": step,
                        "epoch": epoch,
                        "step_train_loss": train_value,
                        "train_time": (datetime.now() - start).total_seconds(),
                        "train_time_per_sample": (
                            datetime.now() - start
                        ).total_seconds()
                        / data.shape[0],
                        "batch_size": data.shape[0],
                        "batch_series_length": data.shape[1],
                        **metrics,
                    },
                )

    val_losses = jnp.zeros((len(val_loader),))
    for idx, batch in enumerate(val_loader):
        if sharding:
            batch = jax.device_put(batch, sharding)
        val_losses = val_losses.at[idx].set(grad_loss_only(mlp, objective, batch))
    return jnp.mean(train_losses), jnp.mean(val_losses), mlp, opt_state


# Function to train the NeuralODE
def train(
    mlp: SurrogateModel,
    opt_state: optax.OptState,
    epochs: list[int],
    fracs: list[float],
    train_loader: SeriesLoader,
    val_loader: SeriesLoader,
    optim: optax.GradientTransformation,
    objective: Objective,
    callbacks: dict[str, list[Callable]] | None = None,
    sharding: jax.sharding.Sharding | None = None,
):
    """Train the NeuralODE

    Args:
        mlp (SurrogateModel): The neural network.
        opt_state (optax.OptState): The state of the optimizer.
        epoch_checkpoints (list[int]): List of epoch checkpoints.
        fracs (list[float]): List of fractions for visual extinctions.
        train_loader (SeriesLoader): The data loader for training data.
        val_loader (SeriesLoader): The data loader for validation data.
        optim (optax.GradientTransformation, optional): The optimizer to use. Defaults to None.
        objective (Objective): The objective that defines the loss terms and their weights.
        callbacks (dict[str, list[Callable]], optional): Callback functions to execute at different parts of each epoch. Defaults to empty.
    """
    if callbacks is None:
        callbacks = {}
    # For training on specific chunks of the dataset to avoid getting caught in local minima
    epoch_checkpoints_a = [1] + list(np.cumsum(epochs, dtype=int)[:-1] + 1)
    epoch_checkpoints_b = list(np.cumsum(epochs, dtype=int))

    train_loss: Scalar = jnp.array(0.0)
    val_loss: Scalar = jnp.array(0.0)
    for frac, epoch_a, epoch_b in zip(
        fracs, epoch_checkpoints_a, epoch_checkpoints_b, strict=True
    ):
        train_loader.set_timeseries_fraction(frac)
        val_loader.set_timeseries_fraction(frac)
        for epoch in range(epoch_a, epoch_b + 1):
            train_loader.shuffle_batches()
            train_loss, val_loss, mlp, opt_state = do_epoch(
                epoch=epoch,
                mlp=mlp,
                optim=optim,
                opt_state=opt_state,
                objective=objective,
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


def _normalization_kwargs(norms: Norms | None) -> dict[str, dict]:
    """PDRLoader keyword arguments for ``norms``; empty dicts make the loader compute them."""
    if norms is None:
        return {
            "independent_variable_normalization_kwargs": {},
            "features_normalization_kwargs": {},
            "auxiliary_features_normalization_kwargs": {},
        }
    return {
        "independent_variable_normalization_kwargs": asdict(norms.iv),
        "features_normalization_kwargs": asdict(norms.data),
        "auxiliary_features_normalization_kwargs": asdict(norms.aux),
    }


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

    # Without a normalisation file, the training loader computes the statistics
    # from the training data; they are then saved and reused for validation.
    normalization_parameters: Norms | None = None
    if conf.normalisations_file and conf.normalisations_file.exists():
        normalization_parameters = read_as(conf.normalisations_file, Norms)
    elif conf.normalisations_file:
        logging.warning(
            f"normalisations_file {conf.normalisations_file!r} not found; "
            "computing the normalisation from the training data"
        )

    # # After parameters are set, get the information of the git repository.
    # conf.update(get_git_info())

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
        **_normalization_kwargs(normalization_parameters),
        batcher=SeriesBatcher(),
        batch_permutation_function=log_semi_sorter,
        use_cache=True,
        batch_subsampling=conf.training_batch_subsampling,
    )
    if normalization_parameters is None:
        normalization_parameters = Norms.from_parameters(
            train_dataloader.get_normalization()
        )
        normalisations_file = save_file_path / "normalisations.toml"
        write_as(normalization_parameters, normalisations_file)
        # Record the file in the saved hyperparameters, so inference uses it too.
        conf = replace(conf, normalisations_file=normalisations_file)
        print(
            f"Saved the normalisation computed from the training data to {normalisations_file}"
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
        **_normalization_kwargs(normalization_parameters),
        batcher=SeriesBatcher(),
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
    plot_freq = int(os.environ.get("NEURALPDR_PLOT_FREQ", 1))
    plot_callback = OneBatchPlotter(save_file_path, plot_freq)
    early_terminate_callback = EarlyTerminate(100, patience=10)
    # neptune_logger = NeptuneLogger(
    #     conf["neptune_project"], conf, tags=conf.get("neptune_tags")
    # )

    callbacks: dict[str, list[Callable]] = {
        "batch_start": [],
        # "batch_end": [neptune_logger],
        "epoch_end": [
            save_weights_callback,
            plot_callback,
            early_terminate_callback,
            # neptune_logger,
        ],
    }

    key = jax.random.PRNGKey(0)
    mlp_key, enc_key, dec_key = jax.random.split(key, 3)
    if conf.checkpoint_file and conf.checkpoint_file.exists():
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
    learning_rate_scheduler = join_schedules(
        learning_rate_scheduler, np.cumsum(boundaries[:-1]).tolist()
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
        optim=optim,
        objective=LatentODEObjective(),
        callbacks=callbacks,
        sharding=sharding,
    )
    return train_loss, val_loss


if __name__ in "__main__":
    conf = get_config()
    main(conf)
