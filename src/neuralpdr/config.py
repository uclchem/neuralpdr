from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal, TypeAlias

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import Field, TypeAdapter

LearningScheduler: TypeAlias = Literal["sgdr", "constant"]
Activation: TypeAlias = Literal["tanh"]


@dataclass
class Split:
    """The splits should normalise to unity."""

    train: float
    validate: float
    test: float


@dataclass
class LearningScheme:
    timeseries_fraction: float
    epochs: int
    lr_scheduler: LearningScheduler
    warmup_epochs: int
    learning_rate: float


@dataclass
class Conf:
    start_index: int
    end_index: int
    batch_size: int
    learning_rate: float
    weight_scale: float
    weight_decay: float
    weight_truncation: float
    enc_dec_depth: int
    enc_dec_width: int
    latent_depth: int
    latent_width: int
    latent_bottleneck: int
    latent_final_activation: Activation
    minimal_timeseries_length: int
    train_test_val_split: Split
    training_batch_subsampling: float
    shuffle_every_n_epochs: int
    aux_features: bool
    save_file_path: Path
    dataset_path: Path
    input_features_file: Path  # FIXME: should this be in config, or data?
    learning_schemes: list[LearningScheme] = Field(default_factory=list)
    double_epochs_last_fraction: bool = False


def read_conf(path: str | Path):
    path = Path(path)
    match path.suffix:
        case ".toml":
            parsed = tomllib.loads(path.read_text())
        case _:
            raise RuntimeError(f"{path.suffix}: unsupported file format {path!r}")

    return TypeAdapter(Conf).validate_python(parsed)
