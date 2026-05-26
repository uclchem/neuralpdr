from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Annotated, Literal, TypeAlias, TypeVar

import tomlkit

from pydantic import Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

LearningScheduler: TypeAlias = Literal["sgdr", "constant"]
Activation: TypeAlias = Literal["tanh", "softplus"]
ModelTypes: TypeAlias = Literal["latent", "fno"]


@dataclass(frozen=True)
class Split:
    """The splits should normalise to unity."""

    train: float
    validate: float
    test: float

    @model_validator(mode="after")
    def normalise(self):
        if abs(self.train + self.validate + self.test - 1) <= 1e-6:
            return self
        else:
            raise ValueError(f"{asdict(self)} does not normalise to 1")


@dataclass(frozen=True)
class LearningScheme:
    lr_scheduler: LearningScheduler
    epochs: int
    learning_rate: float
    warmup_epochs: int = Field(default=0)
    timeseries_fraction: float = Field(default=1.0)


@dataclass(frozen=True)
class _Base:
    start_index: int
    end_index: int
    batch_size: int
    weight_scale: float
    weight_decay: float
    weight_truncation: float
    enc_dec_depth: int
    enc_dec_width: int
    aux_features: bool
    save_file_path: Path
    dataset_path: Path
    input_features_file: Path
    depth: int
    width: int


@dataclass(frozen=True)
class _Train:
    minimal_timeseries_length: int
    train_test_val_split: Split
    training_batch_subsampling: float
    shuffle_every_n_epochs: int
    learning_schemes: list[LearningScheme]
    normalisations_file: Path
    checkpoint_file: Path
    checkpoint_epoch: int  # maybe needs a `None` default


@dataclass(frozen=True)
class _Latent:
    bottleneck: int
    final_activation: Activation


@dataclass(frozen=True)
class Latent(_Base, _Train, _Latent):
    double_epochs_last_fraction: bool = False
    model: Literal["latent"] = "latent"


@dataclass(frozen=True)
class _FNO:
    spectrum: list[float]


@dataclass(frozen=True)
class FNO(_Base, _Train, _FNO):
    double_epochs_last_fraction: bool = False
    model: Literal["fno"] = "fno"


Conf: TypeAlias = Annotated[Latent | FNO, Field(discriminator="model")]


def _read_toml(path: Path) -> dict:
    match path.suffix:
        case ".toml":
            return tomlkit.loads(path.read_text())
        case ".json":
            return json.loads(path.read_text())
        case _:
            raise RuntimeError(f"{path.suffix}: unsupported file format {path!r}")


def read_conf(path: str | Path) -> Conf:
    if not (path := Path(path)).exists():
        raise RuntimeError(f"{path}: missing config file")
    parsed = _read_toml(path)
    return TypeAdapter(Conf).validate_python(parsed)


def write_conf(config: Conf, path: str | Path):
    adapter = TypeAdapter(Conf)
    data = adapter.dump_python(config, mode="json", exclude_none=True)
    path = Path(path)
    path.write_text(tomlkit.dumps(data))


@dataclass(frozen=True)
class Features:
    iv: str
    data: list[str]
    aux: list[str]


@dataclass(frozen=True)
class IVNorm:
    eps: float = Field(default=0.0)
    mean: float = Field(default=0.0)
    std: float = Field(default=0.0)


@dataclass(frozen=True)
class AUXNorm:
    eps: list[float] = Field(default_factory=lambda: [0.0])
    mean: list[float] = Field(default_factory=lambda: [0.0])
    std: list[float] = Field(default_factory=lambda: [0.0])


@dataclass(frozen=True)
class DataNorm:
    eps: float = Field(default=0.0)
    mean: list[float] = Field(default_factory=lambda: [0.0])
    std: list[float] = Field(default_factory=lambda: [0.0])


@dataclass(frozen=True)
class Norms:
    iv: IVNorm
    aux: AUXNorm
    data: DataNorm


@dataclass(frozen=True)
class DataMetadata:
    train_indices: list[str]
    val_indices: list[str]
    test_indices: list[str]


conf_t = TypeVar("conf_t", Features, Norms, DataMetadata)


def read_as(path: str | Path, as_type: type[conf_t]) -> conf_t:
    if not (path := Path(path)).exists():
        raise RuntimeError(f"{path}: missing {as_type.__name__!r} file")
    parsed = _read_toml(path)
    return TypeAdapter(as_type).validate_python(parsed)
