"""
PDRLoader functionality tests.

These tests verify that the PDRLoader class works correctly for loading and
processing datasets. Dataset structure validation is in test_datasets.py.

Purpose:
    - Test PDRLoader initialization
    - Test data loading into memory
    - Test batch creation
    - Test normalization
    - Test iteration and shuffling
    - Test batching of series (SeriesBatcher)
    - Test that train, validation and test splits share no model

Prerequisites:
    Test datasets should exist (validated by test_datasets.py):
        python scripts/data/create_test_datasets.py

Run tests:
    pytest tests/test_pdrloader.py -v
"""

from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest

from neuralpdr.config import Norms, read_as, write_as
from neuralpdr.data import (
    PDRLoader,
    SeriesBatch,
    SeriesBatcher,
    shuffle_and_split,
    text_from_h5,
)

# Dataset locations - prefer test datasets
TEST_DIR = Path(__file__).parent.parent / "data" / "test"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Use test datasets if available, fall back to full datasets
V1_PATH = (
    TEST_DIR / "3dpdr_dataset_v1_test.h5"
    if (TEST_DIR / "3dpdr_dataset_v1_test.h5").exists()
    else PROCESSED_DIR / "3dpdr_dataset_v1.h5"
)
V2_PATH = (
    TEST_DIR / "3dpdr_dataset_v2_test.h5"
    if (TEST_DIR / "3dpdr_dataset_v2_test.h5").exists()
    else PROCESSED_DIR / "3dpdr_dataset_v2.h5"
)
V3_PATH = (
    TEST_DIR / "3dpdr_dataset_v3_test.h5"
    if (TEST_DIR / "3dpdr_dataset_v3_test.h5").exists()
    else PROCESSED_DIR / "3dpdr_dataset_v3.h5"
)


def get_model_count(dataset_path: Path, max_models: int = 100) -> int:
    """Get number of models available in dataset (up to max_models)."""
    if not dataset_path.exists():
        return 0

    with h5py.File(dataset_path, "r") as f:
        model_keys = [
            k
            for k in f.keys()
            if k not in {"header", "species", "model_ids", "model_df"}
        ]
        return min(len(model_keys), max_models)


class TestV1PDRLoader:
    """PDRLoader tests for v1 dataset."""

    @pytest.fixture
    def v1_config(self):
        """Config for v1 dataset testing."""
        if not V1_PATH.exists():
            pytest.skip(f"v1 dataset not found at {V1_PATH}")

        n_models = get_model_count(V1_PATH, max_models=100)

        return {
            "dataset_path": V1_PATH,
            "independent_variable": "visual_extinction",
            "data_features": [
                "CH3OH",
                "CS",
                "CO+",
                "H2CO",
                "HCO",
                "C2",
                "HCN",
                "NH",
                "HCO+",
                "CN",
                "O2",
                "CH",
                "H2O",
                "C",
                "CO",
                "O",
                "H2",
                "H",
                "e-",
            ],
            "auxiliary_features": [
                "visual_extinction",
                "tgas",
                "tdust",
                "density",
                "radfield",
            ],
            "index_range": (0, 120),
            "model_indices": None,
            "batch_size": 8,
            "load_first_n_keys": n_models,
        }

    def test_v1_initialization(self, v1_config):
        """Test PDRLoader can initialize with v1 dataset."""
        loader = PDRLoader(**v1_config)
        assert loader is not None
        assert len(loader.model_indices) == v1_config["load_first_n_keys"]

    def test_v1_data_loading(self, v1_config):
        """Test data loads correctly into memory."""
        loader = PDRLoader(**v1_config)

        n_expected = v1_config["load_first_n_keys"]
        assert len(loader.feature_data_by_model) == n_expected
        assert len(loader.independent_data_by_model) == n_expected
        assert len(loader.auxiliary_data_by_model) == n_expected

        # Check shapes
        first_model = loader.model_indices[0]
        assert loader.feature_data_by_model[first_model].shape[0] == 120  # time steps
        assert loader.feature_data_by_model[first_model].shape[1] == 19  # species
        assert loader.auxiliary_data_by_model[first_model].shape[1] == 5  # aux features

    def test_v1_batch_creation(self, v1_config):
        """Test batches are created with correct shapes."""
        loader = PDRLoader(**v1_config)

        n_models = v1_config["load_first_n_keys"]
        batch_size = v1_config["batch_size"]
        expected_batches = n_models // batch_size
        assert len(loader) == expected_batches

        # Check batch shapes
        batch = loader[0]
        assert batch.iv.shape[0] == batch_size
        assert batch.data.shape[0] == batch_size
        assert batch.aux.shape[0] == batch_size
        assert batch.data.shape[2] == 19  # species
        assert batch.aux.shape[2] == 5  # aux features

    def test_v1_normalization(self, v1_config):
        """Test normalization parameters are computed."""
        loader = PDRLoader(**v1_config)

        norm_params = loader.get_normalization()
        assert "iv" in norm_params
        assert "data" in norm_params
        assert "aux" in norm_params

        assert norm_params["data"]["mean"].shape == (19,)
        assert norm_params["data"]["std"].shape == (19,)

    def test_v1_normalization_round_trip(self, v1_config, tmp_path):
        """Computed normalisation, saved and read back, normalises the data identically."""
        computed = PDRLoader(**v1_config)
        norms = Norms.from_parameters(computed.get_normalization())
        path = tmp_path / "normalisations.toml"
        write_as(norms, path)
        read_back = read_as(path, Norms)
        assert read_back == norms

        reused = PDRLoader(
            **v1_config,
            independent_variable_normalization_kwargs=asdict(read_back.iv),
            features_normalization_kwargs=asdict(read_back.data),
            auxiliary_features_normalization_kwargs=asdict(read_back.aux),
        )
        # Computed statistics are float64 arrays, read-back ones Python floats, so
        # the float32 data is normalised in float64 and float32 respectively.
        for name in ("independent_series", "feature_series", "auxiliary_series"):
            for key, expected in getattr(computed, name).items():
                np.testing.assert_allclose(
                    getattr(reused, name)[key], expected, rtol=1e-6, atol=1e-6
                )

    @pytest.mark.parametrize(
        "frac, n_steps",
        [(32, 32), (32.0, 32), (0.5, 60), (1.0, 120)],
    )
    def test_v1_timeseries_fraction(self, v1_config, frac, n_steps):
        """Above 1.0 is a number of steps (also as a float, as pydantic gives it);
        at most 1.0 is a fraction of the 120-step series."""
        loader = PDRLoader(**v1_config)
        loader.set_timeseries_fraction(frac)
        for batch in loader:
            np.testing.assert_array_equal(batch.mask.sum(axis=1), n_steps)

    def test_v1_iteration(self, v1_config):
        """Test iteration over batches."""
        loader = PDRLoader(**v1_config)

        batch_count = 0
        for batch in loader:
            batch_count += 1
            assert batch.iv.shape[0] <= v1_config["batch_size"]
            assert batch.data.shape[0] <= v1_config["batch_size"]
            assert batch.aux.shape[0] <= v1_config["batch_size"]

        assert batch_count == len(loader)


class TestV2PDRLoader:
    """PDRLoader tests for v2 dataset."""

    @pytest.fixture
    def v2_config(self):
        """Config for v2 dataset testing."""
        if not V2_PATH.exists():
            pytest.skip(f"v2 dataset not found at {V2_PATH}")

        n_models = get_model_count(V2_PATH, max_models=100)

        return {
            "dataset_path": V2_PATH,
            "independent_variable": "visual_extinction",
            "data_features": [
                "H3+",
                "He+",
                "H2+",
                "CH5+",
                "CH4+",
                "O+",
                "OH+",
                "C+",
                "H2O+",
                "H3O+",
                "CO+",
                "O2+",
                "CH2",
                "H2O",
                "H+",
                "CH3+",
                "CH",
                "CH3",
                "HCO+",
                "CH2+",
                "C",
                "He",
                "CH+",
                "CO",
                "OH",
                "O",
                "H2",
                "H",
                "e-",
            ],
            "auxiliary_features": [
                "visual_extinction",
                "tgas",
                "tdust",
                "density",
                "radfield",
            ],
            "index_range": (0, 100),
            "model_indices": None,
            "batch_size": 8,
            "load_first_n_keys": n_models,
        }

    def test_v2_initialization(self, v2_config):
        """Test PDRLoader can initialize with v2 dataset."""
        loader = PDRLoader(**v2_config)
        assert loader is not None
        assert len(loader.model_indices) == v2_config["load_first_n_keys"]

    def test_v2_data_loading(self, v2_config):
        """Test data loads correctly."""
        loader = PDRLoader(**v2_config)

        n_expected = v2_config["load_first_n_keys"]
        assert len(loader.feature_data_by_model) == n_expected
        assert len(loader.independent_data_by_model) == n_expected
        assert len(loader.auxiliary_data_by_model) == n_expected

        first_model = loader.model_indices[0]
        assert loader.feature_data_by_model[first_model].shape[1] == 29  # species
        assert loader.auxiliary_data_by_model[first_model].shape[1] == 5  # aux features

    def test_v2_batch_creation(self, v2_config):
        """Test batches are created correctly."""
        loader = PDRLoader(**v2_config)

        n_models = v2_config["load_first_n_keys"]
        batch_size = v2_config["batch_size"]
        expected_batches = n_models // batch_size
        assert len(loader) == expected_batches

        batch = loader[0]
        assert batch.iv.shape[0] == batch_size
        assert batch.data.shape[2] == 29  # species
        assert batch.aux.shape[2] == 5  # aux features

    def test_v2_normalization(self, v2_config):
        """Test normalization parameters."""
        loader = PDRLoader(**v2_config)

        norm_params = loader.get_normalization()
        assert norm_params["data"]["mean"].shape == (29,)
        assert norm_params["data"]["std"].shape == (29,)

    def test_v2_iteration(self, v2_config):
        """Test iteration over batches."""
        loader = PDRLoader(**v2_config)

        batch_count = 0
        for _ in loader:
            batch_count += 1

        assert batch_count == len(loader)


class TestV3PDRLoader:
    """PDRLoader tests for v3 dataset."""

    @pytest.fixture
    def v3_config(self):
        """Config for v3 dataset testing."""
        if not V3_PATH.exists():
            pytest.skip(f"v3 dataset not found at {V3_PATH}")

        n_models = get_model_count(V3_PATH, max_models=100)

        return {
            "dataset_path": V3_PATH,
            "independent_variable": "visual_extinction",
            "data_features": [
                "H3+",
                "He+",
                "H2+",
                "O2",
                "CH5+",
                "CH4+",
                "O+",
                "OH+",
                "C+",
                "CH4",
                "H2O+",
                "H3O+",
                "CO+",
                "O2+",
                "CH2",
                "H2O",
                "H+",
                "CH3+",
                "CH",
                "CH3",
                "HCO+",
                "CH2+",
                "C",
                "He",
                "CH+",
                "CO",
                "OH",
                "O",
                "H2",
                "H",
                "e-",
            ],
            "auxiliary_features": [
                "visual_extinction",
                "tgas",
                "tdust",
                "density",
                "radfield",
            ],
            "index_range": (0, 100),
            "model_indices": None,
            "batch_size": 8,
            "load_first_n_keys": n_models,
        }

    def test_v3_initialization(self, v3_config):
        """Test PDRLoader can initialize with v3 dataset."""
        loader = PDRLoader(**v3_config)
        assert loader is not None
        assert len(loader.model_indices) == v3_config["load_first_n_keys"]

    def test_v3_data_loading(self, v3_config):
        """Test data loads correctly."""
        loader = PDRLoader(**v3_config)

        n_expected = v3_config["load_first_n_keys"]
        assert len(loader.feature_data_by_model) == n_expected
        assert len(loader.independent_data_by_model) == n_expected
        assert len(loader.auxiliary_data_by_model) == n_expected

        first_model = loader.model_indices[0]
        assert loader.feature_data_by_model[first_model].shape[1] == 31  # species
        assert loader.auxiliary_data_by_model[first_model].shape[1] == 5  # aux features

    def test_v3_batch_creation(self, v3_config):
        """Test batches are created correctly."""
        loader = PDRLoader(**v3_config)

        n_models = v3_config["load_first_n_keys"]
        batch_size = v3_config["batch_size"]
        expected_batches = n_models // batch_size
        assert len(loader) == expected_batches

        batch = loader[0]
        assert batch.iv.shape[0] == batch_size
        assert batch.data.shape[2] == 31  # species
        assert batch.aux.shape[2] == 5  # aux features

    def test_v3_normalization(self, v3_config):
        """Test normalization parameters."""
        loader = PDRLoader(**v3_config)

        norm_params = loader.get_normalization()
        assert norm_params["data"]["mean"].shape == (31,)
        assert norm_params["data"]["std"].shape == (31,)

    def test_v3_iteration(self, v3_config):
        """Test iteration over batches."""
        loader = PDRLoader(**v3_config)

        batch_count = 0
        for _ in loader:
            batch_count += 1

        assert batch_count == len(loader)


class TestPDRLoaderFeatures:
    """Tests for PDRLoader utility features."""

    def test_shuffle_batches(self):
        """Test batch shuffling works."""
        if not V1_PATH.exists():
            pytest.skip("v1 dataset not found")

        config = {
            "dataset_path": V1_PATH,
            "independent_variable": "visual_extinction",
            "data_features": ["H2", "CO", "H", "O"],
            "auxiliary_features": ["tgas", "density", "radfield"],
            "index_range": (0, 50),
            "model_indices": None,
            "batch_size": 8,
            "load_first_n_keys": min(get_model_count(V1_PATH), 50),
        }

        loader = PDRLoader(**config)
        first_batch_models = loader.get_batch_keys()[0].copy()

        loader.shuffle_batches()
        second_batch_models = loader.get_batch_keys()[0]

        # Order should be different (with very high probability for >10 models)
        if len(first_batch_models) >= 8:
            assert not all(
                a == b
                for a, b in zip(first_batch_models, second_batch_models, strict=False)
            )

    def test_inv_normalize(self):
        """Test inverse normalization."""
        if not V1_PATH.exists():
            pytest.skip("v1 dataset not found")

        config = {
            "dataset_path": V1_PATH,
            "independent_variable": "visual_extinction",
            "data_features": ["H2", "CO"],
            "auxiliary_features": ["tgas", "density"],
            "index_range": (0, 50),
            "model_indices": None,
            "batch_size": 8,
            "load_first_n_keys": min(get_model_count(V1_PATH), 20),
        }

        loader = PDRLoader(**config)

        batch = loader[0]

        # Test inverse normalization
        denorm_data = loader.inv_normalize(batch.data[0], "data")
        assert denorm_data.shape == batch.data[0].shape


class TestSeriesBatcher:
    """Tests for building SeriesBatch from series of different lengths."""

    @staticmethod
    def series(lengths: list[int], n_features: int) -> list[np.ndarray]:
        return [
            np.arange(length * n_features, dtype=np.float32).reshape(length, n_features)
            for length in lengths
        ]

    def test_padding_and_mask(self):
        """Series are edge-padded to a common length; the mask excludes the padding."""
        lengths = [5, 3, 4]
        iv = [
            np.linspace(0.1, 1.0, length, dtype=np.float32)[:, None]
            for length in lengths
        ]
        batch = SeriesBatcher()(iv, self.series(lengths, 4), self.series(lengths, 2))

        assert isinstance(batch, SeriesBatch)
        assert batch.iv.shape == (3, 5)
        assert batch.data.shape == (3, 5, 4)
        assert batch.aux.shape == (3, 5, 2)
        assert batch.mask.shape == (3, 5)
        assert batch.mask.dtype == bool
        expected_mask = np.arange(5)[None, :] < np.array(lengths)[:, None]
        np.testing.assert_array_equal(batch.mask, expected_mask)
        # Edge padding repeats the last step of each series.
        np.testing.assert_array_equal(batch.iv[1, 3:], [1.0, 1.0])
        np.testing.assert_array_equal(
            batch.data[1, 3:], np.repeat(batch.data[1, 2:3], 2, 0)
        )

    def test_max_length(self):
        """With max_length, each series is cut to a window of at most that many steps."""
        np.random.seed(0)
        lengths = [40, 50]
        iv = [
            np.linspace(0.1, 1.0, length, dtype=np.float32)[:, None]
            for length in lengths
        ]
        batch = SeriesBatcher(max_length=10)(
            iv, self.series(lengths, 3), self.series(lengths, 1)
        )

        assert batch.iv.shape == (2, 10)
        assert batch.mask.all()
        # The window is a contiguous slice of the original series.
        for i in range(2):
            start = int(np.argmin(np.abs(iv[i][:, 0] - batch.iv[i, 0])))
            np.testing.assert_array_equal(batch.iv[i], iv[i][start : start + 10, 0])


class TestSplits:
    """Train, validation and test data must come from different models."""

    def test_splits_share_no_model(self):
        """shuffle_and_split and the loaders built from it keep the splits disjoint."""
        if not V1_PATH.exists():
            pytest.skip("v1 dataset not found")

        model_ids = text_from_h5(V1_PATH, "model_ids")
        splits = shuffle_and_split(
            model_indices=list(model_ids),
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
        )
        assert sorted(sum(splits, [])) == sorted(model_ids)

        loader_ids = [
            set(
                PDRLoader(
                    dataset_path=V1_PATH,
                    independent_variable="visual_extinction",
                    data_features=["H2", "CO"],
                    auxiliary_features=["tgas", "density"],
                    index_range=(0, 50),
                    model_indices=split,
                    batch_size=4,
                    drop_last=False,
                ).model_indices
            )
            for split in splits
        ]
        for ids, split in zip(loader_ids, splits, strict=True):
            assert ids == set(split)
        train_ids, val_ids, test_ids = loader_ids
        assert not train_ids & val_ids
        assert not train_ids & test_ids
        assert not val_ids & test_ids
