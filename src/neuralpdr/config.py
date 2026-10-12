import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from string import Template
from typing import Annotated, Literal, TypeAlias, TypeVar

import numpy as np
import tomlkit
from pydantic import Field, TypeAdapter, field_validator, model_validator
from pydantic.dataclasses import dataclass

LearningScheduler: TypeAlias = Literal["sgdr", "constant"]
Activation: TypeAlias = Literal["tanh", "softplus"]
ModelTypes: TypeAlias = Literal["latent", "fno"]

# Fields whose values should be treated as filesystem paths and resolved at
# config-load time.  Listed in the order they typically appear in TOML files.
_PATH_FIELDS = [
    "input_features_file",
    "normalisations_file",
    "checkpoint_file",
    "dataset_path",
    "save_file_path",
]


def _find_project_root(start: Path) -> Path | None:
    """Walk up the directory tree to find the project root."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return None


def _resolve_path_field(raw: str, config_dir: Path, project_root: Path | None) -> str:
    """Return the best absolute path for a single path-valued config field.

    Resolution order:
    1. Expand ``${VAR}`` environment variables.
    2. Absolute paths are returned as-is.
    3. Relative paths are tried against (a) the config file's directory, then
       (b) the project root.  The first candidate that exists on disk wins.
    4. If neither exists, fall back to the config-dir-relative resolution so
       callers fail with a path that makes sense rather than a CWD-relative one.
    """
    if "${" in raw:
        try:
            raw = Template(raw).substitute(os.environ)
        except KeyError as e:
            raise ValueError(
                f"Environment variable {e} is not set (referenced in config path {raw!r}). "
                "Set the variable or replace it with an absolute path."
            ) from None

    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)

    candidates: list[Path] = [config_dir / p]
    if project_root is not None:
        candidates.append(project_root / p)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return str(resolved)

    # Nothing found on disk — return the config-dir-relative path so downstream
    # errors name a path the user will recognise.
    fallback = candidates[0].resolve()
    logging.debug(
        "Path %r not found relative to config dir or project root; using %s",
        raw,
        fallback,
    )
    return str(fallback)


def _resolve_paths(data: dict, config_file: Path) -> None:
    """Resolve all path fields in a parsed config dict, modifying it in place."""
    config_dir = config_file.parent.resolve()
    project_root = _find_project_root(config_dir)
    for field in _PATH_FIELDS:
        if data.get(field):
            data[field] = _resolve_path_field(
                str(data[field]), config_dir, project_root
            )


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
    end_index: int | None  # None (or -1 in TOML) means use all available timesteps
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

    @field_validator("end_index", mode="before")
    @classmethod
    def _normalise_end_index(cls, v):
        """Convert the TOML sentinel -1 to None ("use all timesteps")."""
        return None if v == -1 else v


@dataclass(frozen=True)
class _Train:
    minimal_timeseries_length: int
    train_test_val_split: Split
    training_batch_subsampling: float
    shuffle_every_n_epochs: int
    learning_schemes: list[LearningScheme]


@dataclass(frozen=True)
class _Latent:
    bottleneck: int
    final_activation: Activation


@dataclass(frozen=True)
class Latent(_Base, _Train, _Latent):
    normalisations_file: Path | None = None
    checkpoint_file: Path | None = None
    checkpoint_epoch: int = 0
    double_epochs_last_fraction: bool = False
    model: Literal["latent"] = "latent"


@dataclass(frozen=True)
class _FNO:
    spectrum: list[float]


@dataclass(frozen=True)
class FNO(_Base, _Train, _FNO):
    normalisations_file: Path | None = None
    checkpoint_file: Path | None = None
    checkpoint_epoch: int = 0
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
    parsed = dict(_read_toml(path))
    _resolve_paths(parsed, config_file=path)
    return TypeAdapter(Conf).validate_python(parsed)


def write_conf(config: Conf, path: str | Path):
    adapter: TypeAdapter = TypeAdapter(Conf)
    data = adapter.dump_python(config, mode="json", exclude_none=True)
    # end_index=None is excluded by exclude_none=True, but TOML has no null type
    # and the field is required, so write back the sentinel value.
    if "end_index" not in data:
        data["end_index"] = -1
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

    @classmethod
    def from_parameters(cls, parameters: dict[str, dict]) -> "Norms":
        """Build from a loader's ``normalization_parameters`` (``{"iv"|"aux"|"data": {"mean", "std", "eps"}}``).

        Scalars and NumPy arrays are converted to the float and list fields of the
        norm classes; a scalar aux ``eps`` is repeated for every aux feature.
        """
        iv, aux, data = parameters["iv"], parameters["aux"], parameters["data"]
        aux_mean = np.atleast_1d(aux["mean"]).tolist()
        return cls(
            iv=IVNorm(
                eps=float(np.squeeze(iv["eps"])),
                mean=float(np.squeeze(iv["mean"])),
                std=float(np.squeeze(iv["std"])),
            ),
            aux=AUXNorm(
                eps=np.broadcast_to(aux["eps"], len(aux_mean)).tolist(),
                mean=aux_mean,
                std=np.atleast_1d(aux["std"]).tolist(),
            ),
            data=DataNorm(
                eps=float(np.squeeze(data["eps"])),
                mean=np.atleast_1d(data["mean"]).tolist(),
                std=np.atleast_1d(data["std"]).tolist(),
            ),
        )


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


def write_as(obj: Features | Norms | DataMetadata, path: str | Path) -> None:
    """Write ``obj`` as TOML so that ``read_as(path, type(obj))`` reads it back."""
    data = TypeAdapter(type(obj)).dump_python(obj, mode="json")
    Path(path).write_text(tomlkit.dumps(data))


def to_json(config: Conf | Features | Norms | DataMetadata) -> bytes:
    adapter = TypeAdapter(type(config))
    data = adapter.dump_python(config, mode="json", exclude_none=True)
    # As in write_conf: end_index=None is excluded by exclude_none=True, but the
    # field is required when the config is read back, so write the sentinel value.
    if isinstance(config, Latent | FNO) and "end_index" not in data:
        data["end_index"] = -1
    return json.dumps(data).encode()
