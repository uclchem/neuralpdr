import json
import os
from pathlib import Path

import equinox as eqx
import neptune
from plot import plot_batch

try:
    from secret_api_key import NEPTUNE_API_TOKEN
except ImportError:
    NEPTUNE_API_TOKEN = None


class SaveWeightCallback:
    def __init__(self, savepath, hyperparameters, frequency=50):
        self.savepath = Path(savepath)
        self.hyperparameters = hyperparameters
        self.frequency = frequency
        with open(self.savepath / "hyperparameters.json", "wb") as fh:
            fh.write(json.dumps(self.hyperparameters).encode("UTF-8"))

    def __call__(self, epoch, model, train_loss, val_loss, train_loader, val_loader):
        if epoch % self.frequency == 0:
            with open(self.savepath / f"weights_epoch_{epoch}.eqx", "wb") as fh:
                eqx.tree_serialise_leaves(fh, model)


class NeptuneLogger:
    def __init__(self, neptune_project_name, hyperparameters, name=None, tags=None):
        self.neptune_project_name = neptune_project_name
        if NEPTUNE_API_TOKEN:
            self.run = neptune.init_run(
                project=self.neptune_project_name,
                api_token=NEPTUNE_API_TOKEN,
                source_files="src/*.py",
                name=name,
                tags=tags,
                monitoring_namespace="monitoring",  # This is the namespace for the monitoring metrics
            )
        else:
            self.run = {}
            self.run["train_loss"] = []
            self.run["val_loss"] = []
        self.run["hyperparameters"] = hyperparameters

    def __call__(self, epoch, model, train_loss, val_loss, train_loader, val_loader):
        self.run["train_loss"].append(train_loss)
        self.run["val_loss"].append(val_loss)

    def __close__(self):
        neptune.stop()


class OneBatchPlotter:
    def __init__(self, save_path, frequency=50):
        self.save_path = save_path
        self.frequency = frequency

    def __call__(self, epoch, model, train_loss, val_loss, train_loader, val_loader):
        if epoch % self.frequency == 0:
            plot_batch(
                model, train_loader, epoch, self.save_path / "train", n_samples=16
            )
            plot_batch(model, val_loader, epoch, self.save_path / "val", n_samples=16)


class EarlyTerminate:
    def __init__(self, threshold, patience=10, expected_improvement=0.001):
        self.high_loss_counter = 0
        self.no_improvement_counter = 0
        self.threshold = threshold
        self.halt_training = False
        self.patience = patience
        self.history = []
        self.expected_improvement = expected_improvement

    def __call__(self, epoch, model, train_loss, val_loss, train_loader, val_loader):
        # Remove runs with high losses due to instabilities
        if val_loss > self.threshold:
            self.high_loss_counter += 1
        elif self.high_loss_counter > 0:
            self.high_loss_counter -= 1
        if self.high_loss_counter > self.patience:
            self.halt_training = True
        # Remove runs with no improvement
        if len(self.history) > self.patience:
            minimum_improvement = (1 - self.expected_improvement) * max(
                self.history[-self.patience :]
            )
            if val_loss > minimum_improvement:
                self.no_improvement_counter += 1
            else:
                self.no_improvement_counter = 0
            if self.no_improvement_counter > self.patience:
                self.halt_training = True
        self.history.append(val_loss)

    def get_stopcondition(self) -> bool:
        return self.halt_training
