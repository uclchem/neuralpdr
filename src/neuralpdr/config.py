from dataclasses import asdict
from pathlib import Path
import sys
from typing import Annotated, Literal, TypeAlias

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

LearningScheduler: TypeAlias = Literal["sgdr", "constant"]
Activation: TypeAlias = Literal["tanh"]
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
    minimal_timeseries_length: int
    train_test_val_split: Split
    training_batch_subsampling: float
    shuffle_every_n_epochs: int
    aux_features: bool
    save_file_path: Path
    dataset_path: Path
    input_features_file: Path
    learning_schemes: list[LearningScheme]


@dataclass(frozen=True)
class Latent(_Base):
    depth: int
    width: int
    bottleneck: int
    final_activation: Activation
    double_epochs_last_fraction: bool = False
    model: Literal["latent"] = "latent"


@dataclass(frozen=True)
class FNO(_Base):
    depth: int
    width: int
    spectrum: list[float]
    double_epochs_last_fraction: bool = False
    model: Literal["fno"] = "fno"


Conf: TypeAlias = Annotated[Latent | FNO, Field(discriminator="model")]


def _read_toml(path: Path) -> dict:
    match path.suffix:
        case ".toml":
            return tomllib.loads(path.read_text())
        case _:
            raise RuntimeError(f"{path.suffix}: unsupported file format {path!r}")


def read_conf(path: str | Path):
    path = Path(path)
    parsed = _read_toml(path)
    return TypeAdapter(Conf).validate_python(parsed)


@dataclass(frozen=True)
class Features:
    iv: str
    data: list[str]
    aux: list[str]


def read_feature_list(path: Path):
    if not path.exists():
        raise RuntimeError(f"{path}: missing features file")
    parsed = _read_toml(path)
    return TypeAdapter(Features).validate_python(parsed)
