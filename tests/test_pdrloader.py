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

Prerequisites:
    Test datasets should exist (validated by test_datasets.py):
        python scripts/data/create_test_datasets.py

Run tests:
    pytest tests/test_pdrloader.py -v
"""

from pathlib import Path

import h5py
import pytest

from neuralpdr.data import PDRLoader, pad_and_stack

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
            "collate_fn": pad_and_stack,
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
        iv_batch, data_batch, aux_batch = loader[0]
        assert iv_batch.shape[0] == batch_size
        assert data_batch.shape[0] == batch_size
        assert aux_batch.shape[0] == batch_size
        assert data_batch.shape[2] == 19  # species
        assert aux_batch.shape[2] == 5  # aux features

    def test_v1_normalization(self, v1_config):
        """Test normalization parameters are computed."""
        loader = PDRLoader(**v1_config)

        norm_params = loader.get_normalization()
        assert "iv" in norm_params
        assert "data" in norm_params
        assert "aux" in norm_params

        assert norm_params["data"]["mean"].shape == (19,)
        assert norm_params["data"]["std"].shape == (19,)

    def test_v1_iteration(self, v1_config):
        """Test iteration over batches."""
        loader = PDRLoader(**v1_config)

        batch_count = 0
        for iv_batch, data_batch, aux_batch in loader:
            batch_count += 1
            assert iv_batch.shape[0] <= v1_config["batch_size"]
            assert data_batch.shape[0] <= v1_config["batch_size"]
            assert aux_batch.shape[0] <= v1_config["batch_size"]

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
            "collate_fn": pad_and_stack,
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

        iv_batch, data_batch, aux_batch = loader[0]
        assert iv_batch.shape[0] == batch_size
        assert data_batch.shape[2] == 29  # species
        assert aux_batch.shape[2] == 5  # aux features

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
        for iv_batch, data_batch, aux_batch in loader:
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
            "collate_fn": pad_and_stack,
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

        iv_batch, data_batch, aux_batch = loader[0]
        assert iv_batch.shape[0] == batch_size
        assert data_batch.shape[2] == 31  # species
        assert aux_batch.shape[2] == 5  # aux features

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
        for iv_batch, data_batch, aux_batch in loader:
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
            "collate_fn": pad_and_stack,
        }

        loader = PDRLoader(**config)
        first_batch_models = loader.get_batch_keys()[0].copy()

        loader.shuffle_batches()
        second_batch_models = loader.get_batch_keys()[0]

        # Order should be different (with very high probability for >10 models)
        if len(first_batch_models) >= 8:
            assert not all(
                a == b for a, b in zip(first_batch_models, second_batch_models)
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
            "collate_fn": pad_and_stack,
        }

        loader = PDRLoader(**config)

        iv_batch, data_batch, aux_batch = loader[0]

        # Test inverse normalization
        denorm_data = loader.inv_normalize(data_batch[0], "data")
        assert denorm_data.shape == data_batch[0].shape
