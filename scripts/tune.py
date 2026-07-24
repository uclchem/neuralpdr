"""Hyperparameter tuning for neuralpdr using Ray Tune + SMAC3.

Loads a base TOML config (e.g. configs/v4/base.toml), overrides a handful of
architecture/optimization hyperparameters per trial, and trains each variant
via ``neuralpdr.train.main``. Hyperparameter suggestions come from SMAC3's
Bayesian optimizer via the standalone ``raytune_smac`` package
(https://github.com/GijsVermarien/raytune-smac), installed as part of this
project's ``tune`` extra (``pip install -e '.[tune]'``).

Usage:
    python scripts/tune.py configs/v4/base.toml --n-trials 200 --max-concurrent 3 \
        --cpus-per-trial 4 --gpus-per-trial 0.33

    # Fast local smoke test (tiny dataset, 1 epoch, CPU only):
    python scripts/tune.py configs/v4/base.toml --smoke-test
"""

import argparse
import math
from dataclasses import replace
from pathlib import Path

import ConfigSpace as CS
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from raytune_smac import SMACSearch

# `neuralpdr.train` configures the JAX backend as a side effect of import, so
# it (and the JAX/equinox exception types used below) is imported lazily
# inside the trainable, once `JAX_PLATFORM_NAME` has been set for the worker
# process — see `make_trainable`.

from neuralpdr.config import LearningScheme, read_conf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Quantization convention: ConfigSpace>=1.0 dropped the `q` kwarg, so integer
# hyperparameters are defined in units of `step` and scaled back up here
# (mirrors the "_q8" suffix pattern used by the previous tuning scripts).
SEARCH_SPACE = {
    "learning_rate": ("float_log", 1e-4, 1e-1),
    "batch_size_q8": ("int", 1, 8),  # -> 8..64
    "enc_dec_width_q32": ("int", 2, 32),  # -> 64..1024
    "enc_dec_depth": ("int", 2, 6),
    "width_q32": ("int", 2, 32),  # -> 64..1024 (latent ODE net width)
    "depth": ("int", 2, 6),  # latent ODE net depth
    "bottleneck_q8": ("int", 1, 32),  # -> 8..256
    "final_activation": ("cat", ["tanh", "softplus"]),
    "aux_features": ("cat", [True, False]),
    "weight_scale": ("float_log", 1e-3, 1e1),
    "weight_truncation": ("float", 1.0, 20.0),
    "weight_decay": ("float_log", 1e-6, 1e-2),
}


def build_config_space(seed: int | None = None) -> CS.ConfigurationSpace:
    cs = CS.ConfigurationSpace(seed=seed)
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "float_log":
            _, lower, upper = spec
            cs.add(
                CS.UniformFloatHyperparameter(name, lower=lower, upper=upper, log=True)
            )
        elif kind == "float":
            _, lower, upper = spec
            cs.add(
                CS.UniformFloatHyperparameter(name, lower=lower, upper=upper, log=False)
            )
        elif kind == "int":
            _, lower, upper = spec
            cs.add(CS.UniformIntegerHyperparameter(name, lower=lower, upper=upper))
        elif kind == "cat":
            _, choices = spec
            cs.add(CS.CategoricalHyperparameter(name, choices=choices))
        else:
            raise ValueError(f"Unknown search space entry kind: {kind!r}")
    return cs


def config_overrides(sampled: dict) -> dict:
    """Translate a sampled SMAC configuration into neuralpdr config field overrides."""
    return {
        "batch_size": sampled["batch_size_q8"] * 8,
        "enc_dec_width": sampled["enc_dec_width_q32"] * 32,
        "enc_dec_depth": sampled["enc_dec_depth"],
        "width": sampled["width_q32"] * 32,
        "depth": sampled["depth"],
        "bottleneck": sampled["bottleneck_q8"] * 8,
        "final_activation": sampled["final_activation"],
        "aux_features": bool(sampled["aux_features"]),
        "weight_scale": sampled["weight_scale"],
        "weight_truncation": sampled["weight_truncation"],
        "weight_decay": sampled["weight_decay"],
    }


def make_trainable(base_config_path: Path, extra_overrides: dict | None = None):
    extra_overrides = extra_overrides or {}

    def train_one(sampled: dict):
        import equinox as eqx
        import jaxlib
        from neuralpdr import train

        base_conf = read_conf(base_config_path)
        ctx = tune.get_context()
        trial_dir = Path(ctx.get_trial_dir())
        trial_dir.mkdir(parents=True, exist_ok=True)

        base_schemes = extra_overrides.get(
            "learning_schemes", base_conf.learning_schemes
        )
        learning_schemes = [
            replace(scheme, learning_rate=sampled["learning_rate"])
            for scheme in base_schemes
        ]
        overrides = {
            **config_overrides(sampled),
            **{k: v for k, v in extra_overrides.items() if k != "learning_schemes"},
        }
        run_name = (
            f"{base_conf.mlflow_run_name or 'tune'}-{ctx.get_trial_id()}"
            if base_conf.mlflow_experiment_name
            else None
        )
        conf = replace(
            base_conf,
            save_file_path=trial_dir,
            learning_schemes=learning_schemes,
            mlflow_run_name=run_name,
            **overrides,
        )

        try:
            train_loss, val_loss = train.main(conf)
        except (jaxlib.xla_extension.XlaRuntimeError, eqx._errors.EqxRuntimeError) as e:
            print(
                f"Trial failed with a JAX/XLA runtime error, reporting a high loss: {e}"
            )
            return {"val_loss": 1e6, "train_loss": 1e6}

        train_loss = float(train_loss)
        val_loss = float(val_loss)
        if not math.isfinite(train_loss):
            train_loss = 1e6
        if not math.isfinite(val_loss):
            val_loss = 1e6
        return {"train_loss": train_loss, "val_loss": val_loss}

    return train_one


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=str, help="Path to the base TOML config to tune around"
    )
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--cpus-per-trial", type=float, default=4)
    parser.add_argument("--gpus-per-trial", type=float, default=0.33)
    parser.add_argument("--storage-path", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="neuralpdr_tune")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny local check (2 trials, 1 epoch, CPU) to validate the pipeline",
    )
    return parser.parse_args()


def main():
    import os

    args = parse_args()
    base_config_path = Path(args.config).resolve()

    n_trials = args.n_trials
    max_concurrent = args.max_concurrent
    cpus_per_trial = args.cpus_per_trial
    gpus_per_trial = args.gpus_per_trial
    storage_path = args.storage_path or str(PROJECT_ROOT / "data" / "results" / "tune")
    extra_overrides = {}
    # Propagated to each trial's worker process, since `neuralpdr.train` reads
    # JAX_PLATFORM_NAME as an import-time side effect.
    worker_env_vars = {"JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME", "gpu")}

    if args.smoke_test:
        worker_env_vars["JAX_PLATFORM_NAME"] = "cpu"
        worker_env_vars["CUDA_VISIBLE_DEVICES"] = ""
        worker_env_vars["NEURALPDR_PLOT_FREQ"] = "999999"
        n_trials = 2
        max_concurrent = 1
        cpus_per_trial = 1
        gpus_per_trial = 0
        storage_path = str(PROJECT_ROOT / "data" / "results" / "tune_smoke_test")

        version = base_config_path.parent.name  # e.g. configs/v4/base.toml -> "v4"
        test_dataset = (
            PROJECT_ROOT / "data" / "test" / f"3dpdr_dataset_{version}_test.h5"
        )
        if not test_dataset.exists():
            raise FileNotFoundError(
                f"--smoke-test needs a small test dataset at {test_dataset}"
            )
        extra_overrides = {
            "dataset_path": test_dataset,
            "shuffle_every_n_epochs": 0,
            # Force a tiny model regardless of what SMAC samples, so the
            # smoke test compiles and runs quickly on CPU.
            "enc_dec_width": 32,
            "enc_dec_depth": 2,
            "width": 32,
            "depth": 2,
            "bottleneck": 8,
            "batch_size": 8,
            "learning_schemes": [
                LearningScheme(
                    lr_scheduler="constant",
                    epochs=1,
                    learning_rate=1e-3,
                    warmup_epochs=0,
                    timeseries_fraction=1.0,
                )
            ],
        }

    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": worker_env_vars})

    space = build_config_space(seed=args.seed)
    search = SMACSearch(
        space,
        n_trials=n_trials,
        metric="val_loss",
        mode="min",
        seed=args.seed,
    )
    search = ConcurrencyLimiter(search, max_concurrent=max_concurrent)

    trainable = tune.with_resources(
        make_trainable(base_config_path, extra_overrides),
        {"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    )

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            search_alg=search,
            num_samples=n_trials,
            metric="val_loss",
            mode="min",
            max_concurrent_trials=max_concurrent,
        ),
        run_config=tune.RunConfig(
            name=args.experiment_name,
            storage_path=storage_path,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="val_loss", mode="min")
    print("Best config:", best.config)
    print("Best val_loss:", best.metrics.get("val_loss"))


if __name__ == "__main__":
    main()
