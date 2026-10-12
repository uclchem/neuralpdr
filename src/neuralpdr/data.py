import abc
import json
import logging
import pickle
from collections.abc import Callable, Iterator
from functools import reduce
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array
from tqdm import tqdm

from neuralpdr._typing import (
    BatchAux,
    BatchData,
    BatchIV,
    BatchMask,
    SampleSeries,
    SeriesGroup,
)


def text_from_h5(dataset_path: str | Path, key: str) -> list[str]:
    with h5py.File(str(dataset_path), "r") as h5f:
        return [i.decode("utf-8") for i in h5f[key][:]]


def df_from_h5(dataset_path: str | Path, key: str, columns=None) -> pd.DataFrame:
    with h5py.File(str(dataset_path), "r") as h5f:
        return pd.DataFrame(h5f[key][:], columns=columns)


class SeriesBatch(eqx.Module):
    """A batch of whole series, padded to a common length T.

    Attributes:
        iv (BatchIV): The independent variable
        data (BatchData): The data features
        aux (BatchAux): The auxiliary features
        mask (BatchMask): True for the steps that are used in the loss
    """

    iv: BatchIV
    data: BatchData
    aux: BatchAux
    mask: BatchMask


# The batches of a loader: a list of batches, or a single batch whose leaves have a
# leading batch-index dimension N when all batches have the same shape.
Batched: TypeAlias = list[SeriesBatch] | SeriesBatch


class SeriesBatcher:
    """Builds a ``SeriesBatch`` from the series of a group of samples.

    The series are edge-padded to a common length by ``pad_and_stack``. With a
    ``max_length``, a random window of at most that many steps is taken from each
    series first. The mask marks the steps where the padded independent variable
    changes, which excludes the edge padding (and the first step if the
    independent variable starts at exactly 1.0).
    """

    def __init__(self, max_length: int | None = None) -> None:
        """
        Args:
            max_length (int, optional): The maximal number of steps per series; all
                steps when None. Defaults to None.
        """
        self.max_length = max_length

    def __call__(
        self,
        iv: list[SampleSeries],
        data: list[SampleSeries],
        aux: list[SampleSeries],
    ) -> SeriesBatch:
        """Pad and stack the series of a group of samples into a batch.

        Args:
            iv (list[SampleSeries]): The independent variable of each sample
            data (list[SampleSeries]): The data features of each sample
            aux (list[SampleSeries]): The auxiliary features of each sample

        Returns:
            SeriesBatch: The batch
        """
        batch_data, batch_aux, batch_iv = pad_and_stack(
            data, aux, iv, random_sample_number=self.max_length
        )
        mask = jnp.diff(batch_iv, axis=1, prepend=1.0) != 0.0
        return SeriesBatch(
            iv=batch_iv[:, :, 0], data=batch_data, aux=batch_aux, mask=mask[:, :, 0]
        )


# Either array type, returned as the same type.
A = TypeVar("A", np.ndarray, Array)


class SeriesLoader(abc.ABC):
    """Interface that the training, inference and plotting code rely on.

    A loader holds a dataset of samples, each consisting of series along one
    independent variable (time, depth, ...): the independent variable itself,
    the data features and the auxiliary features (see ``SeriesGroup``). It
    serves them as normalised, padded batches. How samples are stored on disk
    and normalised is up to the implementation.
    """

    # Identifiers of the samples, in the order they are batched.
    model_indices: list[str]
    # Names of the data features, in column order.
    data_features: list[str]

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of batches."""

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> SeriesBatch:
        """Return the batch at index ``idx``."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[SeriesBatch]:
        """Iterate over all batches."""

    @abc.abstractmethod
    def get_all_batches(self) -> Batched:
        """Return all batches."""

    @abc.abstractmethod
    def get_batch_keys(self) -> list[list[str]]:
        """Return the sample identifiers in each batch."""

    @abc.abstractmethod
    def shuffle_batches(self) -> None:
        """Reshuffle the samples into new batches."""

    @abc.abstractmethod
    def set_timeseries_fraction(self, frac: float | int) -> None:
        """Limit the batches to a fraction (<= 1.0) or a number of steps (> 1.0) of each series."""

    @abc.abstractmethod
    def inv_normalize(self, data: A, series_key: SeriesGroup) -> A:
        """Transform normalised data of the given group back to original units."""


class PDRLoader(SeriesLoader):
    def __init__(
        self,
        dataset_path: Path,
        independent_variable: str,
        data_features: list[str],
        auxiliary_features: list[str],
        index_range: tuple[int, int | None],
        model_indices: list[str] | Literal["all"] | None,
        model_df: pd.DataFrame | None = None,
        batch_size: int = 16,
        stage: str = "",
        independent_variable_normalization_kwargs: dict | None = None,
        features_normalization_kwargs: dict | None = None,
        auxiliary_features_normalization_kwargs: dict | None = None,
        batch_permutation_function: Callable | None = None,
        batcher: SeriesBatcher | None = None,
        load_first_n_keys: int | None = None,
        drop_last: bool = True,
        use_cache: bool = False,
        batch_subsampling: int | float | None = None,
    ) -> None:
        """Dataloader for the PDR dataset

        Args:
            dataset_path (Path): Path to the dataset.
            independent_variable (str): The independent variable to be used.
            features (list[str]): List of features to be loaded.
            auxiliary_features (list[str]): List of auxiliary features to be loaded.
            index_range (tuple[int]): Range of samples to load from timeseries.
            model_indices (list[str] | "all" | None): The models to load: a non-empty list of
                model ids, or None or "all" for every model in the dataset (the first
                ``load_first_n_keys`` if given). An empty list raises a ValueError, so that an
                empty split never silently loads every model.
            model_df (pd.DataFrame, optional): DataFrame with all samples listed. Defaults to None.
            independent_variable_normalization_kwargs (dict, optional): kwargs for normalizing the independent variable. Defaults to {}.
            features_normalization_kwargs (dict, optional): kwargs for normalizing the features. Defaults to {}.
            auxiliary_features_normalization_kwargs (dict, optional): kwargs for normalizing the auxiliary features. Defaults to {}.
            batch_permutation_function (callable, optional): Function to permute the samples into batches. Defaults to None.
            batcher (SeriesBatcher, optional): Builds the batches from the series of the samples. Defaults to SeriesBatcher().
            load_first_n_keys (int, optional): Number of keys to load. Defaults to None.
            drop_last (bool, optional): Drop the last batch if it is smaller than the batch size. Defaults to True.
            use_cache (bool, optional): Use a cache to store the loaded data. Defaults to False.
        """
        self.dataset_path = dataset_path
        self.independent_variable = independent_variable
        self.data_features = data_features
        self.auxilary_features = auxiliary_features
        self.model_df = model_df
        self.start_index, self.end_index = index_range
        self.batch_size = batch_size
        self.normalization_parameters: dict[str, dict[str, float | np.ndarray]] = {}
        self.stage = stage
        if independent_variable_normalization_kwargs is None:
            independent_variable_normalization_kwargs = {}
        if features_normalization_kwargs is None:
            features_normalization_kwargs = {}
        if auxiliary_features_normalization_kwargs is None:
            auxiliary_features_normalization_kwargs = {}
        self.key_template = "{model}/pdr"
        self.batch_permutation_function = batch_permutation_function
        self.batcher = batcher if batcher is not None else SeriesBatcher()
        self.load_first_n_keys = load_first_n_keys
        self.drop_last = drop_last
        self.use_cache = use_cache
        self.batch_subsampling = batch_subsampling
        self.model_indices_per_batch: list[list[str]] = []

        if model_indices is None or model_indices == "all":
            self.model_indices = text_from_h5(dataset_path, "model_ids")
            if self.load_first_n_keys:
                self.model_indices = self.model_indices[: self.load_first_n_keys]
        elif not model_indices:
            # An empty split (e.g. a split fraction of 0.0) must not fall back to all
            # models, or it would leak models of the other splits into this one.
            raise ValueError(
                f"{stage or 'loader'}: model_indices is empty; pass None or 'all' to load "
                "every model in the dataset"
            )
        else:
            self.model_indices = list(model_indices)
        # if self.subsample_function:
        #     self.model_indices = self.subsample_function(dataset_path, self.model_indices)
        self.data_header: list[str] = text_from_h5(dataset_path, "header")
        self.index_independent_variable: list[int] = self.get_indices_from_header(
            self.data_header, [independent_variable]
        )
        self.indices_auxilary_features: list[int] = self.get_indices_from_header(
            self.data_header, auxiliary_features
        )
        self.indices_features: list[int] = self.get_indices_from_header(
            self.data_header, data_features
        )

        self.n_aux_features = len(self.indices_auxilary_features)
        self.n_data_features = len(self.indices_features)

        # Create dicts that store the data in memory, to allow for easy reshuffling on the fly.
        self.feature_data_by_model: dict[str, SampleSeries] = {}
        self.independent_data_by_model: dict[str, SampleSeries] = {}
        self.auxiliary_data_by_model: dict[str, SampleSeries] = {}

        # The batches: a list, or one batch with a leading batch-index dimension
        self.batches: Batched = []
        # Store all the names of each of the samples in a list
        self.batched_indices: list[list] = []

        # Load all the data into memory
        self.load_data()

        # When end_index is None we use all available timesteps; derive the
        # series length from the loaded data so set_timeseries_fraction works.
        if self.end_index is None:
            # feature_data_by_model is loaded in load_data()
            self.timeseries_length = max(
                len(v) for v in self.feature_data_by_model.values()
            )
        else:
            self.timeseries_length = self.end_index - self.start_index

        # Apply the normalization to the data
        # TODO: add saved normalisation parameters.
        self.independent_series = self.normalize(
            self.independent_data_by_model,
            type_="iv",
            **independent_variable_normalization_kwargs,
        )
        self.feature_series = self.normalize(
            self.feature_data_by_model, type_="data", **features_normalization_kwargs
        )
        self.auxiliary_series = self.normalize(
            self.auxiliary_data_by_model,
            type_="aux",
            **auxiliary_features_normalization_kwargs,
        )

        # Create the batches
        self.shuffle_batches()

    @staticmethod
    def get_indices_from_header(header, feature_names):
        """Get the indices of the features in the header

        Args:
            data_header (list[str]): List of with all names in the header
            feature_names (list[str]): List of feature names to get the indices for

        Returns:
            list[int]: List of indices of the features in the header in sorted order.
        """
        try:
            indices = [header.index(feature) for feature in feature_names]
        except ValueError as e:
            logging.error(f"Could not find find a feature in the header: {header}")
            raise e
        # Check if there is just one index, or if the indices are in increasing order
        if len(indices) == 1 or all(np.diff(indices) > 0):
            return indices
        else:
            logging.warning(
                "The indices are not in increasing order, they will be sorted and disregard the original order"
            )
            indices = sorted(indices)
            return indices

    def load_data(self):
        """Load data from disk into memory."""
        NEED_TO_LOAD = True
        # Load the data from the dataset
        cache_path = Path(
            f"{self.dataset_path.with_suffix('')}_{self.stage}_{len(self.model_indices)}.pickle"
        )
        if self.use_cache:
            if cache_path.exists():
                for _ in tqdm(range(1)):
                    print("Trying to use cache files")
                    # TODO: add try-except block here.
                    with open(cache_path, "rb") as fh:
                        pickle_dict = pickle.load(fh)
                        model_indices = pickle_dict["model_indices"]
                        if model_indices != self.model_indices:
                            msg = "The model indices in the cache do not match the model indices in the dataset"
                            raise RuntimeError(msg)
                        self.independent_data_by_model = pickle_dict[
                            "independent_data_by_model"
                        ]
                        self.feature_data_by_model = pickle_dict[
                            "feature_data_by_model"
                        ]
                        self.auxiliary_data_by_model = pickle_dict[
                            "auxiliary_data_by_model"
                        ]
                        NEED_TO_LOAD = False
        if NEED_TO_LOAD:
            with h5py.File(self.dataset_path, "r") as fh:
                for model_idx in tqdm(self.model_indices):
                    # Load the timeseries data from the models)
                    data = fh[self.key_template.format(model=model_idx)][:]
                    features = data[
                        self.start_index : self.end_index, self.indices_features
                    ]
                    auxilary_features = data[
                        self.start_index : self.end_index,
                        self.indices_auxilary_features,
                    ]
                    independent_variable = data[
                        self.start_index : self.end_index,
                        self.index_independent_variable,
                    ]
                    # Replace NaNs with 0.0, which will be padded to a minimum value since all data is log-transformed later.
                    features[np.isnan(features)] = 0.0
                    auxilary_features[np.isnan(auxilary_features)] = 0.0
                    if any(np.isnan(independent_variable)):
                        raise ValueError(
                            f"Independent variable has NaNs for model {model_idx}"
                        )
                    # Store the data in memory
                    self.independent_data_by_model[model_idx] = independent_variable
                    self.feature_data_by_model[model_idx] = features
                    self.auxiliary_data_by_model[model_idx] = auxilary_features
        if self.use_cache:
            if not cache_path.exists():
                with open(cache_path, "wb") as fh:
                    pickle.dump(
                        {
                            "model_indices": self.model_indices,
                            "independent_data_by_model": self.independent_data_by_model,
                            "feature_data_by_model": self.feature_data_by_model,
                            "auxiliary_data_by_model": self.auxiliary_data_by_model,
                        },
                        fh,
                    )
                print("Wrote cache files")
            else:
                print("Cache files already exist, not overwriting them")

    def normalize(
        self,
        dataset: dict[str, SampleSeries],
        type_: SeriesGroup,
        mean: float | np.ndarray | None = None,
        std: float | np.ndarray | None = None,
        eps: float | np.ndarray = 1e-20,
    ):
        """Add a small epsilon, log transform it and then standardize it

        Args:
            dataset (dict[str, SampleSeries]): Dictionary of numpy arrays to normalize
            type_ (str, optional): Choose between normalizing "data", "aux" or "iv".
            mean (float | np.ndarray, optional): Mean for standardization, must either be scalar or the same shape as the last dimension of the data
            std (float| np.ndarray, optional): Standard deviation for standardization, must either be scalar or the same shape as the last dimension of the data
            eps (float| np.ndarray, optional): Small epsilon to add to the data. Defaults to 1e-20.
        """
        if (mean is None) != (std is None):
            raise RuntimeError("Either both mean and std must be provided or neither.")

        if mean is None and std is None:
            statistics_shape: dict[str, tuple[int, int] | int] = {
                "data": (len(dataset), self.n_data_features),
                "aux": (len(dataset), self.n_aux_features),
                "iv": (len(dataset)),
            }
            sample_lengths: np.ndarray = np.zeros(len(dataset))
            means: np.ndarray = np.zeros(statistics_shape[type_])
            vars_: np.ndarray = np.zeros(statistics_shape[type_])
            print("Computing statistics for normalization")
            data: np.ndarray
            for idx, data in tqdm(enumerate(dataset.values())):
                sample_lengths[idx] = len(data)
                data = np.log10(data + eps)
                # Compute the statistics, but mask the data at the lower boundary.
                means[idx], vars_[idx] = (
                    np.mean(np.ma.masked_values(data, eps), axis=0),  # type: ignore[arg-type]
                    np.var(np.ma.masked_values(data, eps), axis=0),  # type: ignore[arg-type]
                )
                # numpy type hints for masked_values is incomplete, leading to errors
            # TODO: take the weighted mean and variance here.
            mean = np.average(means, axis=0, weights=sample_lengths)
            # Approximate the standard deviation over each features by adding the variance of the means and the mean of the variances.
            std = np.sqrt(np.mean(vars_, axis=0) + np.var(means, axis=0))
            if std.any() < 1e-30:
                raise ValueError(
                    f"Standard deviation cannot be 0, it is for indices {np.where(std == 1e-30)}"
                )

        assert (mean is not None) and (std is not None), "For the type checker"

        # Save the mean and std for later use
        self.normalization_parameters[type_] = {"mean": mean, "std": std, "eps": eps}
        # Apply the transformation to the data
        for key in dataset:
            dataset[key] = (np.log10(dataset[key] + eps) - mean) / std
        return dataset

    def get_normalization(self) -> dict[str, dict[str, float | np.ndarray]]:
        """Get the normalization parameters for the data and av

        Returns:
            dict[str, dict[str, float | np.ndarray]]: Dictionary with the normalization parameters
        """
        return self.normalization_parameters

    def inv_normalize(self, data: A, series_key: SeriesGroup) -> A:
        """Inverts the normalization process for the given group of series.

        Args:
            data (np.ndarray | Array): Normalised data
            series_key (SeriesGroup): Which group the data belongs to: "iv", "data" or "aux"

        Returns:
            np.ndarray | Array: The data in original units, of the same array type as ``data``
        """
        mean = np.array(self.normalization_parameters[series_key]["mean"])
        std = np.array(self.normalization_parameters[series_key]["std"])
        return data * std + mean

    def get_data(
        self,
    ) -> tuple[
        dict[str, SampleSeries], dict[str, SampleSeries], dict[str, SampleSeries]
    ]:
        """Return the data and av dictionaries

        Returns:
            tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: Tuple with the data and av dictionaries
        """
        return self.independent_series, self.feature_series, self.auxiliary_series

    def create_batches(self, model_indices: list[str] | None = None) -> None:
        """Create the batches of data and av"""
        if model_indices is None:
            model_indices = self.model_indices
        logging.debug(f"Creating new batches for {len(model_indices)} series")
        # Create the batches, including a final batch that is smaller than the batch size
        fenceposts = list(range(0, len(model_indices), self.batch_size))
        if not self.drop_last:
            fenceposts += [len(model_indices)]
        # Create the list of grouped model indices for the batches
        batch_indices_lil = [
            model_indices[start:stop]
            for start, stop in zip(fenceposts[:-1], fenceposts[1:], strict=True)
        ]
        # Ensure we shuffle the batches before load the data to ensure
        # the batches are not loaded in increasing series length.
        batch_indices_lil = [
            batch_indices_lil[i]
            for i in np.random.permutation(range(len(batch_indices_lil)))
        ]
        self.model_indices_per_batch = batch_indices_lil

        # Iterate over the grouped model indices and create the batches
        batches = [
            self.batcher(
                [self.independent_data_by_model[idx] for idx in batch_indices],
                [self.feature_data_by_model[idx] for idx in batch_indices],
                [self.auxiliary_data_by_model[idx] for idx in batch_indices],
            )
            for batch_indices in batch_indices_lil
        ]

        # If all batches have the same shape, stack them into one batch of arrays:
        if batches and all(batch.iv.shape == batches[0].iv.shape for batch in batches):
            self.batches = jax.tree.map(lambda *leaves: jnp.stack(leaves), *batches)
        else:
            self.batches = batches

    def get_all_batches(self) -> Batched:
        """Get all the batches

        Returns:
            Batched: A list of batches, or one batch with a leading batch-index dimension
        """
        return self.batches

    def get_batch_keys(self) -> list[list[str]]:
        """Get the keys of the batches

        Returns:
            list[list]: List of keys for the batches
        """
        return self.model_indices_per_batch

    def set_timeseries_fraction(self, frac: float | int) -> None:
        """Set the fraction of the timeseries to load

        Args:
            frac (float | int): Fraction of each series (at most 1.0), or the number
                of steps per series (above 1.0; 32 and 32.0 are the same)
        """
        print("trying to set the next fraction index with frac", frac)
        if frac == 1.0 or frac is None:
            self.batcher.max_length = None
        elif frac > 1.0:
            # Values above 1.0 are an absolute series length cap, not a
            # fraction of the timeseries length. Pydantic coerces TOML ints
            # to float on LearningScheme.timeseries_fraction, so `frac` may
            # arrive as e.g. 32.0 rather than 32 -- treat it the same way.
            self.batcher.max_length = int(frac)
        elif isinstance(frac, (float, int)):
            # Set the end index to the fraction of the timeseries length
            self.batcher.max_length = np.ceil(frac * self.timeseries_length).astype(int)
        else:
            raise ValueError("Fraction must be either a float or an integer")
        # Load the data again
        self.shuffle_batches()

    def shuffle_batches(self) -> None:
        """Shuffle the batches of data and av"""
        logging.debug("Shuffling the batches of data and av")
        if self.batch_permutation_function is not None:
            self.model_indices = self.batch_permutation_function(
                self.model_indices, self.feature_data_by_model
            )
        else:
            self.model_indices = np.random.permutation(self.model_indices).tolist()
        if self.batch_subsampling:
            _batch_size: int | list[int]
            if isinstance(self.batch_subsampling, float):
                _batch_size = int(len(self.model_indices) * self.batch_subsampling)
            elif isinstance(self.batch_subsampling, int):
                _batch_size = [self.batch_subsampling]
            else:
                raise ValueError(
                    "Batch subsampling must be either a float or an integer"
                )
            model_indices: list[str] = np.random.choice(
                self.model_indices, size=_batch_size, replace=False
            ).tolist()
            # Reload the data with the sub batching
            self.create_batches(model_indices)
        else:
            self.create_batches()

    def __len__(self) -> int:
        """Return the number of batches

        Returns:
            int: Number of batches
        """
        if isinstance(self.batches, list):
            return len(self.batches)
        return len(self.batches.iv)

    def __getitem__(self, idx: int) -> SeriesBatch:
        """Get the batch at a given batch index

        Args:
            idx (int): Index of the batch

        Returns:
            SeriesBatch: The batch
        """
        if isinstance(self.batches, list):
            return self.batches[idx]
        return jax.tree.map(lambda leaf: leaf[idx], self.batches)

    def __iter__(self) -> Iterator[SeriesBatch]:
        return (self[idx] for idx in range(len(self)))


def shuffle_and_split(
    *,
    df: pd.DataFrame = None,
    model_indices: list | None = None,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> tuple[list[str], list[str], list[str]]:
    """Function to shuffle and split the dataset into training, validation and test sets.

    Args:
        df (pd.DataFrame): The dataframe with the sample names in the index
        train_split (float): Fraction of the dataset to use for training
        val_split (float): Fraction of the dataset to use for validation
        test_split (float): Fraction of the dataset to use for testing

    Returns:
        tuple[list[str], list[str], list[str]]: The list of model indices for the training, validation and test sets
    """
    if df is None and model_indices is None:
        msg = "Either a dataframe or a list of model indices must be provided"
        raise RuntimeError(msg)
    if df is not None:
        model_indices = df.index.to_list()
    assert model_indices
    np.random.seed(1234)
    np.random.shuffle(model_indices)
    num_models = len(model_indices)
    border1 = int(num_models * train_split)
    border2 = int(num_models * (train_split + val_split))
    if df is not None:
        return df.iloc[0:border1], df.iloc[border1:border2], df.iloc[border2:]
    else:
        return (
            model_indices[0:border1],
            model_indices[border1:border2],
            model_indices[border2:],
        )


def save_split(savepath: Path, split: list[str]) -> None:
    with open(savepath, "w") as f:
        json.dump(split, f)


def load_split(savepath: Path) -> list[str]:
    with open(savepath) as f:
        split = json.load(f)
    return split


def pad_and_stack(
    *batches: list[SampleSeries], random_sample_number: int | None = None
) -> list[Array]:
    """Pad and stack the batch of data

    Args:
        *batches (list[SampleSeries]): Lists of variable length series to pad and stack

    Returns:
        list[Array]: Stacked numpy array
    """
    if not reduce(lambda i, j: j if i == j else False, map(len, batches)):
        msg = "All arrays in the batch must have the same length"
        raise ValueError(msg)

    lengths = np.array([len(data) for data in batches[0]], dtype=int)
    if random_sample_number is not None and random_sample_number > 0:
        random_starts_flt = np.random.uniform(size=lengths.shape[0])
        random_starts = np.floor(
            random_starts_flt * (lengths - random_sample_number)
        ).astype(int)
        random_starts[random_starts < 0] = 0
        random_ends = np.minimum(random_starts + random_sample_number, lengths).astype(
            int
        )
        batches = tuple(
            [
                series[a:b]
                for series, a, b in zip(batch, random_starts, random_ends, strict=True)
            ]
            for batch in batches
        )
        max_length = max([len(data) for data in batches[0]])
    else:
        max_length = max(lengths)
    # Padding function:
    max_length = max_length if max_length <= 32 else int(np.ceil(max_length / 8) * 8)
    padded_batches = [
        jnp.array(
            np.stack(
                [
                    np.pad(data, ((0, max_length - len(data)), (0, 0)), mode="edge")
                    for data in _batch
                ]
            )
        )
        for _batch in batches
    ]
    return padded_batches


def log_semi_sorter(
    model_indices: list[str],
    feature_data_by_model: dict[str, SampleSeries],
    log_noise_parameter: float = 0.01,
):
    lengths = np.array([len(feature_data_by_model[model]) for model in model_indices])
    semi_random_sort_key = np.log10(lengths) + np.random.uniform(
        -log_noise_parameter, log_noise_parameter, lengths.shape
    )
    sorted_indices_dict: dict[str, float] = dict(
        zip(model_indices, semi_random_sort_key, strict=True)
    )
    sorted_indices = sorted(sorted_indices_dict, key=sorted_indices_dict.__getitem__)
    sorted_indices = list(sorted_indices)
    return sorted_indices


def filter_models_by_series_length(
    length: int, model_indices: list[str], dataset_path: str | Path
):
    with h5py.File(str(dataset_path), "r") as fh:
        model_indices = [
            model for model in model_indices if fh[model + "/pdr"].shape[0] >= length
        ]
    return model_indices
