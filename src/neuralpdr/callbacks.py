import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import mlflow
import orbax.checkpoint as ocp

from .config import to_json
from .plot import plot_batch


def _flatten(value, prefix=""):
    """Flatten a (possibly nested) dataclass/dict/list into flat str-valued params.

    mlflow.log_params only accepts flat scalar values, unlike Neptune's
    namespace-based nesting, so dicts/dataclasses/lists are recursively
    expanded into dotted/indexed keys.
    """
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    flat = {}
    if isinstance(value, dict):
        for k, v in value.items():
            flat.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            flat.update(_flatten(v, f"{prefix}.{i}" if prefix else str(i)))
    else:
        flat[prefix] = value
    return flat


class SaveWeightCallback:
    def __init__(self, savepath, hyperparameters, frequency=1):
        self.savepath = Path(savepath)
        self.hyperparameters = hyperparameters
        self.frequency = frequency
        with open(self.savepath / "hyperparameters.json", "wb") as fh:
            fh.write(to_json(self.hyperparameters))

    def __call__(self, **kwargs):
        epoch = kwargs["epoch"]
        model = kwargs["model"]
        if epoch % self.frequency == 0:
            with open(self.savepath / f"weights_epoch_{epoch}.eqx", "wb") as fh:
                eqx.tree_serialise_leaves(fh, model)
        self.frequency += 1


class MlflowLogger:
    def __init__(
        self, experiment_name, hyperparameters, tracking_uri=None, name=None, tags=None
    ):
        self.active = experiment_name is not None
        if self.active:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run(run_name=name, tags=tags)
            mlflow.log_params(_flatten(hyperparameters))
        else:
            self.run = None

    def __call__(self, **kwargs):
        step = kwargs.get("step", kwargs.get("epoch"))
        self.log_metrics(kwargs["mlflow_metrics"], step=step)

    def log_metric(self, key, value, step=None):
        if self.active:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics, step=None):
        if self.active:
            mlflow.log_metrics(metrics, step=step)

    def get_client(self):
        return mlflow if self.active else {}

    def __close__(self):
        if self.active:
            mlflow.end_run()


class JaxProfiler:
    def __init__(self, profile_path, start_epoch, end_epoch, start_step, end_step):
        self.profile_path = profile_path
        assert start_epoch < end_epoch, "Start epoch must be less than end epoch"
        assert start_step < end_step, "Start step must be less than end step"
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, **kwargs):
        epoch = kwargs["epoch"]
        step = kwargs["step"]
        if epoch == self.start_epoch and step == self.start_step:
            jax.profiler.start_trace(self.trace_path, create_perfetto_trace=True)
        elif epoch == self.end_epoch and step == self.end_step:
            jax.profiler.stop_trace()

    def __close__(self):
        jax.profiler.stop_trace()


class CudaartProfiler:
    # "/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0/lib64/libcudart.so"
    def __init__(self, libcudart_path, start_epoch, end_epoch, start_step, end_step):
        from ctypes import cdll

        self.libcudart = cdll.LoadLibrary(libcudart_path)
        assert start_epoch < end_epoch, "Start epoch must be less than end epoch"
        assert start_step < end_step, "Start step must be less than end step"
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, **kwargs):
        epoch = kwargs["epoch"]
        step = kwargs["step"]
        if epoch == self.start_epoch and step == self.start_step:
            self.libcudart.cudaProfilerStart()
        if epoch == self.end_epoch and step == self.end_step:
            self.libcudart.cudaProfilerStop()

    def __close__(self):
        self.libcudart.cudaProfilerStop()


class LogModelWeightNorms:
    def __init__(self, mlflow_logger: MlflowLogger):
        self.mlflow_logger = mlflow_logger

    @staticmethod
    def is_linear(x):
        return isinstance(x, eqx.nn.Linear)

    def get_weights(self, m):
        return [
            x.weight
            for x in jax.tree_util.tree_leaves(m, is_leaf=self.is_linear)
            if self.is_linear(x)
        ]

    def __call__(self, **kwargs):
        model = kwargs["model"]
        epoch = kwargs.get("epoch")
        for i, weights in enumerate(self.get_weights(model)):
            self.mlflow_logger.log_metric(
                f"weights_layer{i}_l2", float(jnp.mean(weights**2)), step=epoch
            )


class OneBatchPlotter:
    def __init__(self, save_path, frequency=50):
        self.save_path = save_path
        self.frequency = frequency

    def __call__(self, **kwargs):
        epoch = kwargs["epoch"]
        if epoch % self.frequency == 0:
            logging.debug("Plotting the model")
            plot_batch(
                kwargs["model"],
                kwargs["train_loader"],
                epoch,
                self.save_path / "train",
                n_samples=8,
            )
            plot_batch(
                kwargs["model"],
                kwargs["val_loader"],
                epoch,
                self.save_path / "val",
                n_samples=8,
            )
            logging.debug("Model plotted")


class EarlyTerminate:
    def __init__(self, threshold, patience=10, expected_improvement=0.001):
        self.high_loss_counter = 0
        self.no_improvement_counter = 0
        self.threshold = threshold
        self.halt_training = False
        self.patience = patience
        self.history = []
        self.expected_improvement = expected_improvement

    def __call__(self, epoch, **kwargs) -> bool:
        val_loss = kwargs["val_loss"]
        # Remove runs with high losses due to instabilities
        if val_loss > self.threshold:
            logging.debug(f"High loss counter +1: {self.high_loss_counter}")
            self.high_loss_counter += 1
        elif self.high_loss_counter > 0:
            logging.debug(f"High loss counter -1: {self.high_loss_counter}")
            self.high_loss_counter -= 1
        else:
            logging.debug(f"High loss counter: {self.high_loss_counter}")
        if self.high_loss_counter > self.patience:
            logging.debug(f"High loss counterexceeds patience ({self.patience})")
            self.halt_training = True
        # Remove runs with no improvement
        if len(self.history) > self.patience:
            minimum_improvement = (1 - self.expected_improvement) * max(
                self.history[-self.patience :]
            )
            if val_loss > minimum_improvement:
                self.no_improvement_counter += 1
                logging.debug(
                    f"No improvement counter +1: {self.no_improvement_counter}"
                )
            else:
                logging.debug(
                    f"No improvement counter reset to 0: {self.no_improvement_counter}"
                )
                self.no_improvement_counter = 0
            if self.no_improvement_counter > self.patience:
                logging.debug(
                    f"No improvement counter exceeds patience ({self.patience})"
                )
                self.halt_training = True
        self.history.append(val_loss)
        return self.halt_training


class Checkpointer:
    def __init__(self, save_dir, save_interval_steps=200):
        self.counter = 0
        self.saveable = None
        self.static = None
        self.save_interval_steps = save_interval_steps
        self.save_path = ocp.test_utils.erase_and_create_empty(save_dir)
        options = ocp.CheckpointManagerOptions(
            max_to_keep=10, save_interval_steps=self.save_interval_steps
        )
        self.checkpointer = ocp.CheckpointManager(self.save_path, options=options)

    def __call__(self, epoch, step, **kwargs):
        train_loss = kwargs["train_loss"]
        model = kwargs["model"]
        if jnp.isnan(train_loss):
            # Reload the old checkpoint
            logging.warning(
                f"Detected a NaN in epoch {epoch}, reloading checkpoint: {self.checkpointer.latest_step()}"
            )
            self.checkpointer.wait_until_finished()
            saveable, self.static = eqx.partition(model, eqx.is_array_like)
            # Reload the one but latest checkpoint
            restored = self.checkpointer.restore(
                self.checkpointer.latest_step() - self.save_interval_steps,
                args=ocp.args.StandardRestore(saveable),
            )
            model = eqx.combine(restored, self.static)
        else:
            # Save the checkpoints
            saveable, self.static = eqx.partition(model, eqx.is_array_like)
            self.checkpointer.save(self.counter, args=ocp.args.StandardSave(saveable))
            self.counter += 1
        return model
