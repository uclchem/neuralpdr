from functools import reduce
import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Union

import h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm


def text_from_h5(dataset_path: str | Path, key: str) -> list[str]:
    with h5py.File(str(dataset_path), "r") as h5f:
        return [i.decode("utf-8") for i in h5f[key][:]]


def df_from_h5(dataset_path: str | Path, key: str, columns=None) -> pd.DataFrame:
    with h5py.File(str(dataset_path), "r") as h5f:
        return pd.DataFrame(h5f[key][:], columns=columns)


class PDRLoader:
    def __init__(
        self,
        dataset_path: Path,
        independent_variable: str,
        data_features: list[str],
        auxiliary_features: list[str],
        index_range: tuple[int, int],
        model_indices: list[str],
        model_df: pd.DataFrame = None,
        batch_size: int = 16,
        stage: str = "",
        independent_variable_normalization_kwargs: dict = {},
        features_normalization_kwargs: dict = {},
        auxiliary_features_normalization_kwargs: dict = {},
        batch_permutation_function: callable = None,
        collate_fn: callable = lambda x: np.stack(x, axis=0),
        load_first_n_keys: int = None,
        drop_last: bool = True,
        use_cache: bool = False,
        batch_subsampling: Union[int, float] = None,
    ) -> None:
        """Dataloader for the PDR dataset

        Args:
            dataset_path (Path): Path to the dataset.
            independent_variable (str): The independent variable to be used.
            features (list[str]): List of features to be loaded.
            auxiliary_features (list[str]): List of auxiliary features to be loaded.
            index_range (tuple[int]): Range of samples to load from timeseries.
            model_indices (list[str]): List of model indices to load.
            model_df (pd.DataFrame, optional): DataFrame with all samples listed. Defaults to None.
            independent_variable_normalization_kwargs (dict, optional): kwargs for normalizing the independent variable. Defaults to {}.
            features_normalization_kwargs (dict, optional): kwargs for normalizing the features. Defaults to {}.
            auxiliary_features_normalization_kwargs (dict, optional): kwargs for normalizing the auxiliary features. Defaults to {}.
            batch_permutation_function (callable, optional): Function to permute the samples into batches. Defaults to None.
            collate_fn (callable, optional): Function to collate data into batches. Defaults to lambda x: np.stack(x, axis=0).
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
        self.timeseries_length = self.end_index - self.start_index
        self.batch_size = batch_size
        self.model_indices = model_indices
        self.normalization_parameters = {}
        self.stage = stage
        self.key_template = "{model}/pdr"
        self.batch_permutation_function = batch_permutation_function
        self.collate_fn = collate_fn
        self.load_first_n_keys = load_first_n_keys
        self.drop_last = drop_last
        self.use_cache = use_cache
        self.batch_subsampling = batch_subsampling
        self.model_indices_per_batch = []

        self.dynamic_end_index = None

        if not self.model_indices:
            self.model_indices = text_from_h5(dataset_path, "model_ids")
            if self.load_first_n_keys:
                self.model_indices = self.model_indices[: self.load_first_n_keys]
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
        self.feature_data_by_model: dict[str, np.array] = {}
        self.independent_data_by_model: dict[str, np.array] = {}
        self.auxiliary_data_by_model: dict[str, np.array] = {}

        # Create lists to store the batched data, or an array of jax data with shape (n_batches, batch_size, av_points, n_features)
        self.batched_feature_data: list[np.array] = []
        self.batched_independent_data: list[np.array] = []
        self.batched_auxiliary_data: list[np.array] = []
        # Store all the names of each of the samples in a list
        self.batched_indices: list[list] = []

        # Load all the data into memory
        self.load_data()

        # Apply the normalization to the data
        # TODO: add saved normalisation parameters.
        self.independent_series = self.normalize(
            self.independent_data_by_model,
            type="iv",
            **independent_variable_normalization_kwargs,
        )
        self.feature_series = self.normalize(
            self.feature_data_by_model, type="data", **features_normalization_kwargs
        )
        self.auxiliary_series = self.normalize(
            self.auxiliary_data_by_model,
            type="aux",
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
                for a in tqdm((range(1))):
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
        dataset: list[np.ndarray],
        type: str,
        mean: float | np.ndarray | None = None,
        std: float | np.ndarray | None = None,
        eps: float | np.ndarray | None = 1e-20,
    ):
        """Add a small epsilon, log transform it and then standardize it

        Args:
            dataset (list[np.ndarray]): List of numpy arrays to normalize
            type (str, optional): Choose between normalizing "data", "aux" or "iv".
            mean (float | np.ndarray, optional): Mean for standardization, must either be scalar or the same shape as the last dimension of the data
            std (float| np.ndarray, optional): Standard deviation for standardization, must either be scalar or the same shape as the last dimension of the data
            eps (float| np.ndarray, optional): Small epsilon to add to the data. Defaults to 1e-20.
        """
        if (mean is None and std is not None) or (mean is not None and std is None):
            raise RuntimeError("Either both mean and std must be provided or neither.")

        if mean is None:
            statistics_shape = {
                "data": (len(dataset), self.n_data_features),
                "aux": (len(dataset), self.n_aux_features),
                "iv": (len(dataset)),
            }
            sample_lengths = np.zeros(len(dataset))
            means = np.zeros(statistics_shape[type])
            vars = np.zeros(statistics_shape[type])
            print("Computing statistics for normalization")
            for idx, data in tqdm(enumerate(dataset.values())):
                sample_lengths[idx] = len(data)
                data = np.log10(data + eps)
                # Compute the statistics, but mask the data at the lower boundary.
                means[idx], vars[idx] = (
                    np.mean(np.ma.masked_values(data, eps), axis=0),
                    np.var(np.ma.masked_values(data, eps), axis=0),
                )
            # TODO: take the weighted mean and variance here.
            mean = np.average(means, axis=0, weights=sample_lengths)
            # Approximate the standard deviation over each features by adding the variance of the means and the mean of the variances.
            std = np.sqrt(np.mean(vars, axis=0) + np.var(means, axis=0))
            if std.any() < 1e-30:
                raise ValueError(
                    f"Standard deviation cannot be 0, it is for indices {np.where(std == 1e-30)}"
                )
        # Save the mean and std for later use
        self.normalization_parameters[type] = {"mean": mean, "std": std, "eps": eps}
        # Apply the transformation to the data
        for key in dataset:
            dataset[key] = (np.log10(dataset[key] + eps) - mean) / std
        return dataset

    def get_normalization(self) -> dict[str, dict[str, Union[float, np.array]]]:
        """Get the normalization parameters for the data and av

        Returns:
            dict[str, dict[str, Union[float, np.array]]]: Dictionary with the normalization parameters
        """
        return self.normalization_parameters

    def inv_normalize(self, data: np.array, series_key: str) -> np.array:
        """Inverts the normalization process for the visual extinction data.

        Args:
            data (np.array): Array of visual extinction data that is normalized

        Returns:
            np.array:  Visual extinction data is original coordinates
        """
        mean = np.array(self.normalization_parameters[series_key]["mean"])
        std = np.array(self.normalization_parameters[series_key]["std"])
        return data * std + mean

    def get_data(self) -> tuple[dict[str, np.array], dict[str, np.array]]:
        """Return the data and av dictionaries

        Returns:
            tuple[dict[str, np.array], dict[str, np.array]]: Tuple with the data and av dictionaries
        """
        return self.independent_series, self.feature_series, self.auxiliary_series

    def create_batches(self, model_indices=None) -> None:
        """Create the batches of data and av"""
        if model_indices is None:
            model_indices = self.model_indices
        logging.debug(f"Creating new batches for {len(model_indices)} series")
        # reset the batched data and av
        del (
            self.batched_independent_data,
            self.batched_feature_data,
            self.batched_auxiliary_data,
        )
        self.batched_feature_data = []
        self.batched_auxiliary_data = []
        self.batched_independent_data = []
        # Create the batches, including a final batch that is smaller than the batch size
        fenceposts = list(range(0, len(model_indices), self.batch_size))
        if not self.drop_last:
            fenceposts += [
                len(model_indices),
            ]
        # Create the list of grouped model indices for the batches
        batch_indices_lil = [
            self.model_indices[start:stop]
            for start, stop in zip(fenceposts[:-1], fenceposts[1:])
        ]
        # Ensure we shuffle the batches before load the data to ensure
        # the batches are not loaded in increasing series length.
        batch_indices_lil = [
            batch_indices_lil[i]
            for i in np.random.permutation(range(len(batch_indices_lil)))
        ]
        self.model_indices_per_batch = batch_indices_lil

        # Iterate over the grouped model indices and create the batches
        for batch_indices in batch_indices_lil:
            batch_data, batch_aux, batch_iv = self.collate_fn(
                [self.feature_data_by_model[idx] for idx in batch_indices],
                [self.auxiliary_data_by_model[idx] for idx in batch_indices],
                [self.independent_data_by_model[idx] for idx in batch_indices],
                random_sample_number=self.dynamic_end_index,
            )
            # print("inside the batching function the shape is: ", batch_data.shape)
            self.batched_feature_data.append(batch_data)
            self.batched_auxiliary_data.append(batch_aux)
            self.batched_independent_data.append(batch_iv)

        # If all batches have the same shape, cast them into one big array:
        if all(
            (
                [
                    batch_iv.shape == self.batched_independent_data[0].shape
                    for batch_iv in self.batched_independent_data
                ]
            )
        ):
            last_batch_length = len(self.batched_feature_data[-1])
            self.batched_feature_data = jnp.array(
                self.batched_feature_data[
                    : -1 if last_batch_length != self.batch_size else None
                ]
            )
            self.batched_auxiliary_data = jnp.array(
                self.batched_auxiliary_data[
                    : -1 if last_batch_length != self.batch_size else None
                ]
            )
            self.batched_independent_data = jnp.array(
                self.batched_independent_data[
                    : -1 if last_batch_length != self.batch_size else None
                ]
            )

    def get_all_batches(
        self,
    ) -> tuple[Union[list[np.array], jnp.array], Union[list[np.array], jnp.array]]:
        """Get all the batches of data and av

        Returns:
            tuple[Union[list[np.array], jnp.array], Union[list[np.array], jnp.array]]: Tuple with the batched av and data
        """
        return (
            self.batched_independent_data,
            self.batched_feature_data,
            self.batched_auxiliary_data,
        )

    def get_batch_keys(self) -> list[list]:
        """Get the keys of the batches

        Returns:
            list[list]: List of keys for the batches
        """
        return self.model_indices_per_batch

    def set_timeseries_fraction(self, frac: Union[float, int]) -> None:
        """Set the fraction of the timeseries to load

        Args:
            frac (float): Fraction of the timeseries to load
        """
        print("trying to set the next fraction index with frac", frac)
        if frac == 1.0 or frac is None:
            self.dynamic_end_index = None
        if isinstance(frac, float):
            # Set the end index to the fraction of the timeseries length
            self.dynamic_end_index = np.ceil(frac * self.timeseries_length).astype(int)
        elif isinstance(frac, int):
            # Set the end index to the fraction of the timeseries length
            self.dynamic_end_index = frac
        else:
            raise ValueError("Fraction must be either a float or an integer")
        # Load the data again
        self.shuffle_batches()

    def shuffle_batches(self) -> None:
        """Shuffle the batches of data and av"""
        logging.debug(f"Shuffling the batches of data and av")
        if self.batch_permutation_function is not None:
            self.model_indices = self.batch_permutation_function(
                self.model_indices, self.feature_data_by_model
            )
        else:
            self.model_indices = np.random.permutation(self.model_indices)
        if self.batch_subsampling:
            if isinstance(self.batch_subsampling, float):
                model_indices = np.random.choice(
                    self.model_indices,
                    size=int(len(self.model_indices) * self.batch_subsampling),
                    replace=False,
                ).tolist()
            elif isinstance(self.batch_subsampling, int):
                model_indices = np.random.choice(
                    self.model_indices, size=[self.batch_subsampling], replace=False
                )
            else:
                raise ValueError(
                    "Batch subsampling must be either a float or an integer"
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
        return len(self.batched_feature_data)

    def __getitem__(
        self, idx: int
    ) -> tuple[Union[list[np.array], jnp.array], Union[list[np.array], jnp.array]]:
        """Get the batched data and av at a given batch index, returning them as jax arrays or list of arrays

        Args:
            idx (int): Index of the batch

        Returns:
            tuple[Union[list[np.array], jnp.array], Union[list[np.array], jnp.array]]: Tuple with the batched av and data
        """
        return (
            self.batched_independent_data[idx],
            self.batched_feature_data[idx],
            self.batched_auxiliary_data[idx],
        )

    def __iter__(self):
        return iter(
            zip(
                self.batched_independent_data,
                self.batched_feature_data,
                self.batched_auxiliary_data,
            )
        )


def shuffle_and_split(
    *,
    df: pd.DataFrame = None,
    model_indices: list = None,
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
        model_indices = df.index.to_numpy()
    np.random.seed(1234)
    np.random.shuffle(model_indices)
    num_models = len(model_indices)
    border1 = int(num_models * train_split)
    border2 = int(num_models * (train_split + val_split))
    if df:
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
    with open(savepath, "r") as f:
        split = json.load(f)
    return split


class PadAndStack:
    def __init__(self, random_sample_number=None):
        self.random_sample_number = random_sample_number

    def __call__(self, batch):
        return pad_and_stack(batch, self.random_sample_number)


def pad_and_stack(*batches, random_sample_number=None):
    """Pad and stack the batch of data

    Args:
        batch (list[np.array]): List of numpy arrays to stack

    Returns:
        np.array: Stacked numpy array
    """
    if not reduce(lambda i, j: j if i == j else False, map(len, batches)):
        msg = "All arrays in the batch must have the same length"
        raise ValueError(msg)

    lengths = np.array([len(data) for data in batches[0]], dtype=int)
    if random_sample_number is not None and random_sample_number > 0:
        random_starts = np.random.uniform(size=lengths.shape[0])
        random_starts = np.floor(
            random_starts * (lengths - random_sample_number)
        ).astype(int)
        random_starts[random_starts < 0] = 0
        random_ends = np.minimum(random_starts + random_sample_number, lengths).astype(
            int
        )
        batches = [
            [series[a:b] for series, a, b in zip(batch, random_starts, random_ends)]
            for batch in batches
        ]
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


def log_semi_sorter(model_indices, feature_data_by_model, log_noise_parameter=0.01):
    lengths = np.array([len(feature_data_by_model[model]) for model in model_indices])
    semi_random_sort_key = np.log10(lengths) + np.random.uniform(
        -log_noise_parameter, log_noise_parameter, lengths.shape
    )
    sorted_indices = dict(zip(model_indices, semi_random_sort_key))
    sorted_indices = sorted(sorted_indices, key=sorted_indices.get)
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
