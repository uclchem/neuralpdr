"""
Integration tests for PDRLoader with v1, v2, and v3 datasets.

These tests verify that the PDRLoader class can successfully load and process
all three datasets for training, including batch creation and normalization.

Prerequisites:
    - Install all dependencies: pip install -r requirements.txt
    - Process datasets: ./scripts/data/workflow_setup_all.sh
    
Run tests:
    pytest tests/test_pdrloader_integration.py -v
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from neuralpdr.data import PDRLoader, pad_and_stack

# Expected dataset locations
# Prefer minimal test datasets for CI, fall back to full datasets
DATA_DIR = Path(__file__).parent.parent / "data"
TEST_DIR = DATA_DIR / "test"
PROCESSED_DIR = DATA_DIR / "processed"

# Use test datasets if available, otherwise use full datasets
V1_PATH = TEST_DIR / "3dpdr_dataset_v1_test.h5" if (TEST_DIR / "3dpdr_dataset_v1_test.h5").exists() else PROCESSED_DIR / "3dpdr_dataset_v1.h5"
V2_PATH = TEST_DIR / "3dpdr_dataset_v2_test.h5" if (TEST_DIR / "3dpdr_dataset_v2_test.h5").exists() else PROCESSED_DIR / "3dpdr_dataset_v2.h5"
V3_PATH = TEST_DIR / "3dpdr_dataset_v3_test.h5" if (TEST_DIR / "3dpdr_dataset_v3_test.h5").exists() else PROCESSED_DIR / "3dpdr_dataset_v3.h5"


def get_model_count(dataset_path: Path, max_models: int = 100) -> int:
    """
    Get number of models available in dataset (up to max_models).
    
    Args:
        dataset_path: Path to HDF5 dataset
        max_models: Maximum number of models to use
        
    Returns:
        Number of models available (capped at max_models)
    """
    if not dataset_path.exists():
        return 0
    
    with h5py.File(dataset_path, "r") as f:
        model_keys = [k for k in f.keys() if k not in {"header", "species", "model_ids", "model_df"}]
        return min(len(model_keys), max_models)


class TestV1PDRLoader:
    """Integration tests for v1 dataset with PDRLoader."""

    @pytest.fixture
    def v1_config(self):
        """Minimal config for v1 dataset testing."""
        if not V1_PATH.exists():
            pytest.skip(f"v1 dataset not found at {V1_PATH}")
        
        n_models = get_model_count(V1_PATH, max_models=100)
        
        return {
            "dataset_path": V1_PATH,
            "independent_variable": "visual_extinction",
            "data_features": [
                "CH3OH", "CS", "CO+", "H2CO", "HCO", "C2", "HCN", "NH",
                "HCO+", "CN", "O2", "CH", "H2O", "C", "CO", "O", "H2", "H", "e-"
            ],
            "auxiliary_features": [
                "visual_extinction", "tgas", "tdust", "density", "radfield"
                # Note: zeta_init is not available as a time-series auxiliary feature
                # It exists only as a scalar in the auxiliary dataset, not in pdr
            ],
            "index_range": (0, 120),  # Short time series for testing
            "model_indices": None,  # Will load all (limited by load_first_n_keys)
            "batch_size": 8,
            "load_first_n_keys": n_models,  # Adapt to available models
            "collate_fn": pad_and_stack,
        }

    def test_v1_loader_initialization(self, v1_config):
        """Test that PDRLoader can be initialized with v1 dataset."""
        if not V1_PATH.exists():
            pytest.skip(f"v1 dataset not found at {V1_PATH}")

        loader = PDRLoader(**v1_config)
        assert loader is not None
        assert len(loader.model_indices) == v1_config["load_first_n_keys"]

    def test_v1_data_loading(self, v1_config):
        """Test that data is loaded into memory correctly."""
        if not V1_PATH.exists():
            pytest.skip(f"v1 dataset not found at {V1_PATH}")

        loader = PDRLoader(**v1_config)
        
        n_expected = v1_config["load_first_n_keys"]
        # Check that data dictionaries are populated
        assert len(loader.feature_data_by_model) == n_expected
        assert len(loader.independent_data_by_model) == n_expected
        assert len(loader.auxiliary_data_by_model) == n_expected
        
        # Check data shapes for first model (index_range is 0-120, so 120 time steps)
        first_model = loader.model_indices[0]
        assert loader.feature_data_by_model[first_model].shape[0] == 120  # 120 time steps
        assert loader.feature_data_by_model[first_model].shape[1] == 19  # 19 species
        assert loader.auxiliary_data_by_model[first_model].shape[1] == 5  # 5 aux features

    def test_v1_batch_creation(self, v1_config):
        """Test that batches are created with correct shapes."""
        if not V1_PATH.exists():
            pytest.skip(f"v1 dataset not found at {V1_PATH}")

        loader = PDRLoader(**v1_config)
        
        # Check number of batches (n_models / batch_size, drop_last=True)
        n_models = v1_config["load_first_n_keys"]
        batch_size = v1_config["batch_size"]
        expected_batches = n_models // batch_size  # Integer division
        assert len(loader) == expected_batches
        
        # Check batch shapes
        iv_batch, data_batch, aux_batch = loader[0]
        assert iv_batch.shape[0] == 8  # batch_size
        assert data_batch.shape[0] == 8
        assert aux_batch.shape[0] == 8
        assert data_batch.shape[2] == 19  # 19 species
        assert aux_batch.shape[2] == 5  # 5 aux features

    def test_v1_normalization(self, v1_config):
        """Test that normalization parameters are computed."""
        if not V1_PATH.exists():
            pytest.skip(f"v1 dataset not found at {V1_PATH}")

        loader = PDRLoader(**v1_config)
        
        norm_params = loader.get_normalization()
        assert "iv" in norm_params
        assert "data" in norm_params
        assert "aux" in norm_params
        
        # Check that mean and std exist
        assert "mean" in norm_params["data"]
        assert "std" in norm_params["data"]
        assert norm_params["data"]["mean"].shape == (19,)
        assert norm_params["data"]["std"].shape == (19,)

    def test_v1_iteration(self, v1_config):
        """Test that we can iterate over batches."""
        if not V1_PATH.exists():
            pytest.skip(f"v1 dataset not found at {V1_PATH}")

        loader = PDRLoader(**v1_config)
        
        batch_count = 0
        for iv_batch, data_batch, aux_batch in loader:
            batch_count += 1
            assert iv_batch.shape[0] <= 8  # Last batch may be smaller
            assert data_batch.shape[0] <= 8
            assert aux_batch.shape[0] <= 8
        
        assert batch_count == len(loader)


class TestV2PDRLoader:
    """Integration tests for v2 dataset with PDRLoader."""

    @pytest.fixture
    def v2_config(self):
        """Minimal config for v2 dataset testing."""
        if not V2_PATH.exists():
            pytest.skip(f"v2 dataset not found at {V2_PATH}")
        
        n_models = get_model_count(V2_PATH, max_models=100)
        
        return {
            "dataset_path": V2_PATH,
            "independent_variable": "visual_extinction",
            "data_features": [
                'H3+', 'He+', 'H2+', 'CH5+', 'CH4+', 'O+', 'OH+', 'C+',
                'H2O+', 'H3O+', 'CO+', 'O2+', 'CH2', 'H2O', 'H+', 'CH3+',
                'CH', 'CH3', 'HCO+', 'CH2+', 'C', 'He', 'CH+', 'CO', 'OH',
                'O', 'H2', 'H', 'e-'
            ],
            "auxiliary_features": [
                "visual_extinction", "tgas", "tdust", "density", "radfield"
            ],
            "index_range": (0, 100),
            "model_indices": None,
            "batch_size": 8,
            "load_first_n_keys": n_models,
            "collate_fn": pad_and_stack,
        }

    def test_v2_loader_initialization(self, v2_config):
        """Test that PDRLoader can be initialized with v2 dataset."""
        if not V2_PATH.exists():
            pytest.skip(f"v2 dataset not found at {V2_PATH}")

        loader = PDRLoader(**v2_config)
        assert loader is not None
        assert len(loader.model_indices) == v2_config["load_first_n_keys"]

    def test_v2_data_loading(self, v2_config):
        """Test that data is loaded into memory correctly."""
        if not V2_PATH.exists():
            pytest.skip(f"v2 dataset not found at {V2_PATH}")

        loader = PDRLoader(**v2_config)
        
        n_expected = v2_config["load_first_n_keys"]
        assert len(loader.feature_data_by_model) == n_expected
        assert len(loader.independent_data_by_model) == n_expected
        assert len(loader.auxiliary_data_by_model) == n_expected
        
        first_model = loader.model_indices[0]
        assert loader.feature_data_by_model[first_model].shape[1] == 29  # 29 species
        assert loader.auxiliary_data_by_model[first_model].shape[1] == 5  # 5 aux features

    def test_v2_batch_creation(self, v2_config):
        """Test that batches are created with correct shapes."""
        if not V2_PATH.exists():
            pytest.skip(f"v2 dataset not found at {V2_PATH}")

        loader = PDRLoader(**v2_config)
        
        # Check number of batches (n_models / batch_size, drop_last=True)
        n_models = v2_config["load_first_n_keys"]
        batch_size = v2_config["batch_size"]
        expected_batches = n_models // batch_size
        assert len(loader) == expected_batches
        
        iv_batch, data_batch, aux_batch = loader[0]
        assert iv_batch.shape[0] == 8
        assert data_batch.shape[0] == 8
        assert aux_batch.shape[0] == 8
        assert data_batch.shape[2] == 29  # 29 species
        assert aux_batch.shape[2] == 5  # 5 aux features

    def test_v2_normalization(self, v2_config):
        """Test that normalization parameters are computed."""
        if not V2_PATH.exists():
            pytest.skip(f"v2 dataset not found at {V2_PATH}")

        loader = PDRLoader(**v2_config)
        
        norm_params = loader.get_normalization()
        assert "iv" in norm_params
        assert "data" in norm_params
        assert "aux" in norm_params
        
        assert norm_params["data"]["mean"].shape == (29,)
        assert norm_params["data"]["std"].shape == (29,)

    def test_v2_iteration(self, v2_config):
        """Test that we can iterate over batches."""
        if not V2_PATH.exists():
            pytest.skip(f"v2 dataset not found at {V2_PATH}")

        loader = PDRLoader(**v2_config)
        
        batch_count = 0
        for iv_batch, data_batch, aux_batch in loader:
            batch_count += 1
            assert iv_batch.shape[0] <= 8
            assert data_batch.shape[0] <= 8
            assert aux_batch.shape[0] <= 8
        
        assert batch_count == len(loader)


class TestV3PDRLoader:
    """Integration tests for v3 dataset with PDRLoader."""

    @pytest.fixture
    def v3_config(self):
        """Minimal config for v3 dataset testing."""
        if not V3_PATH.exists():
            pytest.skip(f"v3 dataset not found at {V3_PATH}")
        
        n_models = get_model_count(V3_PATH, max_models=100)
        
        return {
            "dataset_path": V3_PATH,
            "independent_variable": "visual_extinction",
            "data_features": [
                "H3+", "He+", "H2+", "O2", "CH5+", "CH4+", "O+", "OH+", "C+",
                "CH4", "H2O+", "H3O+", "CO+", "O2+", "CH2", "H2O", "H+", "CH3+",
                "CH", "CH3", "HCO+", "CH2+", "C", "He", "CH+", "CO", "OH", "O",
                "H2", "H", "e-"
            ],
            "auxiliary_features": [
                "visual_extinction", "tgas", "tdust", "density", "radfield"
            ],
            "index_range": (0, 100),
            "model_indices": None,
            "batch_size": 8,
            "load_first_n_keys": n_models,  # CRITICAL for v3 (300K+ models in full dataset)
            "collate_fn": pad_and_stack,
        }

    def test_v3_loader_initialization(self, v3_config):
        """Test that PDRLoader can be initialized with v3 dataset."""
        if not V3_PATH.exists():
            pytest.skip(f"v3 dataset not found at {V3_PATH}")

        loader = PDRLoader(**v3_config)
        assert loader is not None
        assert len(loader.model_indices) == v3_config["load_first_n_keys"]

    def test_v3_data_loading(self, v3_config):
        """Test that data is loaded into memory correctly."""
        if not V3_PATH.exists():
            pytest.skip(f"v3 dataset not found at {V3_PATH}")

        loader = PDRLoader(**v3_config)
        
        n_expected = v3_config["load_first_n_keys"]
        assert len(loader.feature_data_by_model) == n_expected
        assert len(loader.independent_data_by_model) == n_expected
        assert len(loader.auxiliary_data_by_model) == n_expected
        
        first_model = loader.model_indices[0]
        assert loader.feature_data_by_model[first_model].shape[1] == 31  # 31 species
        assert loader.auxiliary_data_by_model[first_model].shape[1] == 5  # 5 aux features

    def test_v3_batch_creation(self, v3_config):
        """Test that batches are created with correct shapes."""
        if not V3_PATH.exists():
            pytest.skip(f"v3 dataset not found at {V3_PATH}")

        loader = PDRLoader(**v3_config)
        
        # Check number of batches (n_models / batch_size, drop_last=True)
        n_models = v3_config["load_first_n_keys"]
        batch_size = v3_config["batch_size"]
        expected_batches = n_models // batch_size
        assert len(loader) == expected_batches
        
        iv_batch, data_batch, aux_batch = loader[0]
        assert iv_batch.shape[0] == 8
        assert data_batch.shape[0] == 8
        assert aux_batch.shape[0] == 8
        assert data_batch.shape[2] == 31  # 31 species
        assert aux_batch.shape[2] == 5  # 5 aux features

    def test_v3_normalization(self, v3_config):
        """Test that normalization parameters are computed."""
        if not V3_PATH.exists():
            pytest.skip(f"v3 dataset not found at {V3_PATH}")

        loader = PDRLoader(**v3_config)
        
        norm_params = loader.get_normalization()
        assert "iv" in norm_params
        assert "data" in norm_params
        assert "aux" in norm_params
        
        assert norm_params["data"]["mean"].shape == (31,)
        assert norm_params["data"]["std"].shape == (31,)

    def test_v3_iteration(self, v3_config):
        """Test that we can iterate over batches."""
        if not V3_PATH.exists():
            pytest.skip(f"v3 dataset not found at {V3_PATH}")

        loader = PDRLoader(**v3_config)
        
        batch_count = 0
        for iv_batch, data_batch, aux_batch in loader:
            batch_count += 1
            assert iv_batch.shape[0] <= 8
            assert data_batch.shape[0] <= 8
            assert aux_batch.shape[0] <= 8
        
        assert batch_count == len(loader)


class TestPDRLoaderFeatures:
    """Tests for PDRLoader utility features."""

    def test_shuffle_batches(self):
        """Test that batch shuffling works."""
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
            "load_first_n_keys": 50,
            "collate_fn": pad_and_stack,
        }
        
        loader = PDRLoader(**config)
        first_batch_models = loader.get_batch_keys()[0].copy()
        
        # Shuffle and check that order changed
        loader.shuffle_batches()
        second_batch_models = loader.get_batch_keys()[0]
        
        # Order should be different (with very high probability)
        assert not all(a == b for a, b in zip(first_batch_models, second_batch_models))

    def test_inv_normalize(self):
        """Test inverse normalization function."""
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
            "load_first_n_keys": 20,
            "collate_fn": pad_and_stack,
        }
        
        loader = PDRLoader(**config)
        
        # Get a batch
        iv_batch, data_batch, aux_batch = loader[0]
        
        # Test inverse normalization on data
        denorm_data = loader.inv_normalize(data_batch[0], "data")
        assert denorm_data.shape == data_batch[0].shape
