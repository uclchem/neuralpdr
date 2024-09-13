import json
import logging
from pathlib import Path
from typing import Union

import h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Header for physics part of the data
PHYSICS_HEADER = [
    "time_idx",
    "position",
    "visual_extinction",
    "tgas",
    "tdust",
    "etype",
    "density",
    "radfield",
]
# Header for the auxilary data, this data doesn't change over the course of the simulation.
AUXILARY_HEADER = ["radfield_init", "density_init", "zeta_init"]


def h5py_load(dataset_path, key, return_dataframe=False, columns=None, text=False):
    with h5py.File(dataset_path, "r") as f:
        data = f[key][:]
    if text:
        data = [i.decode("utf-8") for i in data]
    if return_dataframe:
        data = pd.DataFrame(data, columns=columns)
    return data


class PDRLoader:
    def __init__(
        self,
        dataset_path: Path,
        model_df: pd.DataFrame,
        index_range: tuple[int],
        model_indices: list[str],
        input_features: list[str],
        batch_size: int = 16,
        stage: str = "",
        data_normalization_kwargs: dict = {},
        av_normalization_kwargs: dict = {},
        jax_mode: bool = True,
    ) -> None:
        """Dataloader for the PDR dataset

        Args:
            dataset_path (Path): Path to the dataset
            model_df (pd.DataFrame): Path to the dataframe with all samples listed
            index_range (tuple[int]): Range of samples to load from timeseries
            model_indices (list[str]): List of model indices to load
            input_features (list[str]): Features that are loaded, any feature ending in _init is considered auxilary
            batch_size (int, optional): Batch size. Defaults to 16.
            stage (str, optional): Name of the stage of the loader. Defaults to "".
            data_normalization_kwargs (dict, optional): kwargs to provide to the data normalization step for the data. Defaults to {}.
            av_normalization_kwargs (dict, optional): karws to provide to the data normalization step for the av. Defaults to {}.
            jax_mode (bool, optional): Whether to use jax arrays, drops the last batch if it is not exactly batch_size in length. Defaults to True.
        """
        self.dataset_path = dataset_path
        self.model_df = model_df
        self.start_index, self.end_index = index_range
        self.timeseries_length = self.end_index - self.start_index
        self.jax_mode = jax_mode
        self.batch_size = batch_size
        self.model_indices = model_indices
        self.input_features = input_features
        self.n_features = 0
        self.log_scale_parameters = {}
        self.stage = stage

        self.data_header = PHYSICS_HEADER + h5py_load(
            dataset_path, "species", text=True
        )
        self.col_index_by_header = {
            header: i for i, header in enumerate(self.data_header)
        }
        logging.debug(f"Column index by header: {self.col_index_by_header}")

        self.auxilary_data = self.model_df.loc[self.model_indices]

        # Create dicts that store the data in memory, to allow for easy reshuffling on the fly.
        self.data = {}
        self.av = {}

        # Create lists to store the batched data, or an array of jax data with shape (n_batches, batch_size, av_points, n_features)
        self.batched_data = []
        self.batched_av = []
        # Store all the names of each of the samples in a list
        self.batched_indices = []

        # Load all the data into memory
        self.load_data()

        # Apply the normalization to the data
        self.av = self.normalize(self.av, type="av", **av_normalization_kwargs)
        self.data = self.normalize(self.data, type="data", **data_normalization_kwargs)

        # Create the batches
        self.create_batches()

    def load_data(self):
        """Load data from disk into memory."""
        # Create a dictionary with the index of each header string in sorted order, excluding the auxilary ones.
        self.col_index_by_header_to_load = dict(
            sorted(
                {
                    self.col_index_by_header[header]: header
                    for header in self.input_features
                    if header not in AUXILARY_HEADER
                }.items()
            )
        )
        # Convert the dict to a list of indices to load, to be used by h5py.
        column_indices_to_load = list(self.col_index_by_header_to_load.keys())

        # Load the data from the dataset
        key_template = "{model}/pdr"
        with h5py.File(self.dataset_path, "r") as fh:
            for model_idx in self.model_indices:
                # Load the timeseries data from the models)
                data = fh[key_template.format(model=model_idx)][
                    self.start_index : self.end_index, column_indices_to_load
                ]
                # Append the auxilary data to the timeseries data
                auxilary_features = [
                    feature
                    for feature in self.input_features
                    if feature in AUXILARY_HEADER
                ]
                # Append the auxilary data to each timestep
                auxilary_data = (
                    np.ones([data.shape[0], len(auxilary_features)])
                    * self.auxilary_data.loc[model_idx, auxilary_features].values
                )
                data = np.concatenate([data, auxilary_data], axis=1)
                # Replace NaNs with 0.0, which will be padded to a minimum value since all data is log-transformed later.
                if np.isnan(data).any():
                    nan_columns = np.where(np.isnan(data))
                    # Just replace NaNs with 0.0 since they are all underflows.
                    data[nan_columns] = 0.0
                # Store the data in memory
                self.data[model_idx] = data
                # Load the visual extinction data
                self.av[model_idx] = fh[key_template.format(model=model_idx)][
                    :, self.col_index_by_header["visual_extinction"]
                ]
            # Save the number of features
            self.n_features = data.shape[1]
            self.loaded_columns = list(self.col_index_by_header_to_load.values()) + auxilary_features

    def normalize(
        self,
        dataset: list[np.array],
        type: str = "data",
        eps: float = 1e-20,
        mean: Union[float, np.array] = None,
        std: Union[float, np.array] = None,
    ):
        """Add a small epsilon, log transform it and then standardize it

        Args:
            dataset (list[np.array]): List of numpy arrays to normalize
            type (str, optional): Choose between normalizing "data" or "av".
            eps (float, optional): Small epsilon to add to the data. Defaults to 1e-20.
            mean (Union[float, np.array], optional): Mean for standardization, must either be scalar or the same shape as the last dimension of the data
            std (float, optional): Standard deviation for standardization, must either be scalar or the same shape as the last dimension of the data

        """
        assert (mean is None) == (
            std is None
        ), "Either both mean and std must be provided or neither."
        if mean is None:
            if type == "data":
                means = np.zeros((len(dataset), self.n_features))
                vars = np.zeros((len(dataset), self.n_features))
            else:
                means = np.zeros(len(dataset))
                vars = np.zeros(len(dataset))
            for idx, data in enumerate(dataset.values()):
                data = np.log10(data + eps)
                # Compute the statistics, but mask the data at the lower boundary.
                means[idx], vars[idx] = (
                    np.mean(np.ma.masked_values(data, eps), axis=0),
                    np.var(np.ma.masked_values(data, eps), axis=0),
                )
            mean = np.mean(means, axis=0)
            # Approximate the standard deviation over each features by adding the variance of the means and the mean of the variances.
            std = np.sqrt(np.mean(vars, axis=0) + np.var(means, axis=0))
        # Save the mean and std for later usen
        self.log_scale_parameters[type] = {"mean": mean, "std": std}
        # Apply the transformation to the data
        for key in dataset:
            dataset[key] = (np.log10(dataset[key] + eps) - mean) / std
        return dataset

    def get_normalization(self) -> dict[str, dict[str, Union[float, np.array]]]:
        """Get the normalization parameters for the data and av

        Returns:
            dict[str, dict[str, Union[float, np.array]]]: Dictionary with the normalization parameters
        """
        return self.log_scale_parameters

    def inv_normalize_av(self, av: np.array) -> np.array:
        """Inverts the normalization process for the visual extinction data.

        Args:
            av (np.array): Array of visual extinction data that is normalized

        Returns:
            np.array:  Visual extinction data is original coordinates
        """
        mean = self.log_scale_parameters["av"]["mean"]
        std = self.log_scale_parameters["av"]["std"]
        return av * std + mean

    def inv_normalize_data(self, data: np.array) -> np.array:
        """Inverts the normalization process for the data.

        Args:
            data (np.array): Array of data that is normalized

        Returns:
            np.array: Data in original coordinates
        """
        mean = self.log_scale_parameters["data"]["mean"]
        std = self.log_scale_parameters["data"]["std"]
        return data * std + mean

    def get_data(self) -> tuple[dict[str, np.array], dict[str, np.array]]:
        """Return the data and av dictionaries

        Returns:
            tuple[dict[str, np.array], dict[str, np.array]]: Tuple with the data and av dictionaries
        """
        return self.av, self.data

    def create_batches(self) -> None:
        """Create the batches of data and av"""
        # reset the batched data and av
        self.batched_data = []
        self.batched_av = []
        # Create the batches, including a final batch that is smaller than the batch size
        fenceposts = list(range(0, len(self.model_indices), self.batch_size)) + [
            len(self.model_indices)
        ]
        # Create the list of grouped model indices for the batches
        batch_indices_lil = [
            self.model_indices[start:stop]
            for start, stop in zip(fenceposts[:-1], fenceposts[1:])
        ]
        # Iterate over the grouped model indices and create the batches
        for batch_indices in batch_indices_lil:
            batch_data = np.array(
                [self.data[idx][: self.end_index] for idx in batch_indices]
            )
            batch_av = np.array(
                [self.av[idx][: self.end_index] for idx in batch_indices]
            )
            self.batched_data.append(batch_data)
            self.batched_av.append(batch_av)
        # Convert the data to jax arrays if in jax mode, dropping the last batch if it is not exactly batch_size in length
        if self.jax_mode:
            last_batch_length = len(self.batched_data[-1])
            self.batched_data = jnp.array(
                self.batched_data[
                    : -1 if last_batch_length != self.batch_size else None
                ]
            )
            self.batched_av = jnp.array(
                self.batched_av[: -1 if last_batch_length != self.batch_size else None]
            )

    def get_all_batches(
        self,
    ) -> tuple[Union[list[np.array], jnp.array], Union[list[np.array], jnp.array]]:
        """Get all the batches of data and av

        Returns:
            tuple[Union[list[np.array], jnp.array], Union[list[np.array], jnp.array]]: Tuple with the batched av and data
        """
        return self.batched_av, self.batched_data

    def set_timeseries_fraction(self, frac: float) -> None:
        """Set the fraction of the timeseries to load

        Args:
            frac (float): Fraction of the timeseries to load
        """
        # Set the end index to the fraction of the timeseries length
        self.end_index = np.ceil(frac * self.timeseries_length).astype(int)
        # Load the data again
        self.create_batches()

    def shuffle_batches(self) -> None:
        """Shuffle the batches of data and av"""
        self.model_indices = np.random.permutation(self.model_indices)
        # Reload the data
        self.create_batches()

    def __len__(self) -> int:
        """Return the number of batches

        Returns:
            int: Number of batches
        """
        return len(self.batched_data)

    def __getitem__(
        self, idx: int
    ) -> tuple[Union[list[np.array], jnp.array], Union[list[np.array], jnp.array]]:
        """Get the batched data and av at a given batch index, returning them as jax arrays or list of arrays

        Args:
            idx (int): Index of the batch

        Returns:
            tuple[Union[list[np.array], jnp.array], Union[list[np.array], jnp.array]]: Tuple with the batched av and data
        """
        return self.batched_av[idx], self.batched_data[idx]


def shuffle_and_split(
    df: pd.DataFrame, train_split: float, val_split: float, test_split: float
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
    model_indices = df.index.to_numpy()
    np.random.seed(1234)
    np.random.shuffle(model_indices)
    num_models = len(model_indices)
    assert train_split + val_split + test_split == 1
    border1 = int(num_models * train_split)
    border2 = int(num_models * (train_split + val_split))
    df_train = df.iloc[0:border1]
    df_val = df.iloc[border1:border2]
    df_test = df.iloc[border2:]
    return df_train, df_val, df_test

def save_split(savepath:Path , split: list[str]) -> None:
    with open(savepath, 'w') as f:
        json.dump(split, f)

def load_split(savepath: Path) -> list[str]:
    with open(savepath, 'r') as f:
        split = json.load(f)
    return split