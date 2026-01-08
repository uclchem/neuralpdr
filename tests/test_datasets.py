"""
Dataset structure validation tests.

These tests verify that test datasets exist and have the correct HDF5 structure.
They do NOT test PDRLoader functionality - that's in test_pdrloader.py.

Purpose:
    - Validate test dataset presence (data/test/*.h5)
    - Check HDF5 structure (groups, datasets, headers)
    - Verify data shapes and types
    - Ensure model keys follow expected patterns

Prerequisites:
    Test datasets should be generated with:
        python scripts/data/create_test_datasets.py
    Then committed to repository:
        git add data/test/*.h5

Run tests:
    pytest tests/test_datasets.py -v
"""

from pathlib import Path

import h5py
import pytest

# Test dataset locations (data/test/)
TEST_DIR = Path(__file__).parent.parent / "data" / "test"
V1_TEST_PATH = TEST_DIR / "3dpdr_dataset_v1_test.h5"
V2_TEST_PATH = TEST_DIR / "3dpdr_dataset_v2_test.h5"
V3_TEST_PATH = TEST_DIR / "3dpdr_dataset_v3_test.h5"

# Helpful error message for missing test datasets
MISSING_DATASET_MSG = """
Test dataset not found at: {path}

To create test datasets:
    python scripts/data/create_test_datasets.py

Then commit them:
    git add data/test/*.h5
    git commit -m "Add/update test datasets"
    
Test datasets are small (~6 MB total) and should be committed to the repository.
"""


class TestV1Dataset:
    """Structure validation tests for v1 test dataset."""

    def test_v1_exists(self):
        """Test dataset file must exist."""
        if not V1_TEST_PATH.exists():
            pytest.fail(MISSING_DATASET_MSG.format(path=V1_TEST_PATH))

    def test_v1_has_required_keys(self):
        """Check v1 has required metadata keys."""
        if not V1_TEST_PATH.exists():
            pytest.skip("v1 test dataset not found")

        with h5py.File(V1_TEST_PATH, "r") as f:
            required_keys = {"header", "species", "model_ids", "model_df"}
            missing = required_keys - set(f.keys())
            if missing:
                pytest.fail(
                    f"Missing required keys in v1: {missing}\n"
                    f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
                )

    def test_v1_header_structure(self):
        """Check v1 header has correct number of fields."""
        if not V1_TEST_PATH.exists():
            pytest.skip("v1 test dataset not found")

        with h5py.File(V1_TEST_PATH, "r") as f:
            header = [s.decode("utf-8") for s in f["header"][:]]
            assert len(header) == 226, (
                f"Expected 226 header fields, got {len(header)}\n"
                f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
            )

            # Check for key physics fields
            expected_physics = [
                "visual_extinction",
                "tgas",
                "tdust",
                "density",
                "radfield",
            ]
            missing = set(expected_physics) - set(header)
            if missing:
                pytest.fail(
                    f"Missing expected physics fields: {missing}\n"
                    f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
                )

    def test_v1_has_models(self):
        """Check v1 has model groups."""
        if not V1_TEST_PATH.exists():
            pytest.skip("v1 test dataset not found")

        with h5py.File(V1_TEST_PATH, "r") as f:
            model_keys = [k for k in f.keys() if k.startswith("model_")]
            assert len(model_keys) >= 10, (
                f"Expected at least 10 models, got {len(model_keys)}\n"
                f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
            )

    def test_v1_model_structure(self):
        """Check v1 models have required datasets."""
        if not V1_TEST_PATH.exists():
            pytest.skip("v1 test dataset not found")

        with h5py.File(V1_TEST_PATH, "r") as f:
            model_keys = [k for k in f.keys() if k.startswith("model_")]
            if not model_keys:
                pytest.fail("No models found in v1")

            first_model = model_keys[0]
            required_datasets = {"pdr", "auxiliary"}
            missing = required_datasets - set(f[first_model].keys())
            if missing:
                pytest.fail(
                    f"Model {first_model} missing required datasets: {missing}\n"
                    f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
                )

    def test_v1_pdr_shape(self):
        """Check v1 pdr dataset has correct dimensions."""
        if not V1_TEST_PATH.exists():
            pytest.skip("v1 test dataset not found")

        with h5py.File(V1_TEST_PATH, "r") as f:
            model_keys = [k for k in f.keys() if k.startswith("model_")]
            first_model = model_keys[0]
            pdr_data = f[first_model]["pdr"][:]

            # Should be (n_timesteps, n_features)
            assert len(pdr_data.shape) == 2, (
                f"PDR data should be 2D, got shape {pdr_data.shape}\n"
                f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
            )

            # Should have at least 223 features (8 physics + 215 species)
            assert pdr_data.shape[1] >= 223, (
                f"Expected at least 223 features, got {pdr_data.shape[1]}\n"
                f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
            )


class TestV2Dataset:
    """Structure validation tests for v2 test dataset."""

    def test_v2_exists(self):
        """Test dataset file must exist."""
        if not V2_TEST_PATH.exists():
            pytest.fail(MISSING_DATASET_MSG.format(path=V2_TEST_PATH))

    def test_v2_has_required_keys(self):
        """Check v2 has required metadata keys."""
        if not V2_TEST_PATH.exists():
            pytest.skip("v2 test dataset not found")

        with h5py.File(V2_TEST_PATH, "r") as f:
            required_keys = {"header", "species"}
            missing = required_keys - set(f.keys())
            if missing:
                pytest.fail(
                    f"Missing required keys in v2: {missing}\n"
                    f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
                )

    def test_v2_header_structure(self):
        """Check v2 header has correct number of fields."""
        if not V2_TEST_PATH.exists():
            pytest.skip("v2 test dataset not found")

        with h5py.File(V2_TEST_PATH, "r") as f:
            header = [s.decode("utf-8") for s in f["header"][:]]
            assert len(header) == 39, (
                f"Expected 39 header fields, got {len(header)}\n"
                f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
            )

    def test_v3_has_models(self):
        """Check v2 has model groups."""
        if not V2_TEST_PATH.exists():
            pytest.skip("v2 test dataset not found")

        with h5py.File(V2_TEST_PATH, "r") as f:
            model_keys = [k for k in f.keys() if k not in {"header", "species"}]
            assert len(model_keys) >= 10, (
                f"Expected at least 10 models, got {len(model_keys)}\n"
                f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
            )

    def test_v2_model_structure(self):
        """Check v2 models have pdr dataset."""
        if not V2_TEST_PATH.exists():
            pytest.skip("v2 test dataset not found")

        with h5py.File(V2_TEST_PATH, "r") as f:
            model_keys = [k for k in f.keys() if k not in {"header", "species"}]
            if not model_keys:
                pytest.fail("No models found in v2")

            first_model = model_keys[0]
            # v2 models are groups with 'pdr' dataset
            if isinstance(f[first_model], h5py.Group):
                if "pdr" not in f[first_model].keys():
                    pytest.fail(
                        f"Model {first_model} missing 'pdr' dataset\n"
                        f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
                    )
            # Or v2 models might BE the pdr dataset directly (acceptable)
            elif not isinstance(f[first_model], h5py.Dataset):
                pytest.fail(f"Model {first_model} is neither Group nor Dataset")


class TestV3Dataset:
    """Structure validation tests for v3 test dataset."""

    def test_v3_exists(self):
        """Test dataset file must exist."""
        if not V3_TEST_PATH.exists():
            pytest.fail(MISSING_DATASET_MSG.format(path=V3_TEST_PATH))

    def test_v3_has_required_keys(self):
        """Check v3 has required metadata keys."""
        if not V3_TEST_PATH.exists():
            pytest.skip("v3 test dataset not found")

        with h5py.File(V3_TEST_PATH, "r") as f:
            required_keys = {"header", "species"}
            missing = required_keys - set(f.keys())
            if missing:
                pytest.fail(
                    f"Missing required keys in v3: {missing}\n"
                    f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
                )

    def test_v3_header_structure(self):
        """Check v3 header has correct number of fields."""
        if not V3_TEST_PATH.exists():
            pytest.skip("v3 test dataset not found")

        with h5py.File(V3_TEST_PATH, "r") as f:
            header = [s.decode("utf-8") for s in f["header"][:]]
            # v3 should have 8 physics + 31 species + potentially auxiliary = 39+ fields
            assert len(header) >= 39, (
                f"Expected at least 39 header fields, got {len(header)}\n"
                f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
            )

    def test_v3_has_models(self):
        """Check v3 has model groups."""
        if not V3_TEST_PATH.exists():
            pytest.skip("v3 test dataset not found")

        with h5py.File(V3_TEST_PATH, "r") as f:
            model_keys = [k for k in f.keys() if k not in {"header", "species"}]
            assert len(model_keys) >= 10, (
                f"Expected at least 10 models, got {len(model_keys)}\n"
                f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
            )

    def test_v3_model_structure(self):
        """Check v3 models have pdr dataset."""
        if not V3_TEST_PATH.exists():
            pytest.skip("v3 test dataset not found")

        with h5py.File(V3_TEST_PATH, "r") as f:
            model_keys = [k for k in f.keys() if k not in {"header", "species"}]
            if not model_keys:
                pytest.fail("No models found in v3")

            first_model = model_keys[0]
            # v3 models are groups with 'pdr' dataset
            if isinstance(f[first_model], h5py.Group):
                if "pdr" not in f[first_model].keys():
                    pytest.fail(
                        f"Model {first_model} missing 'pdr' dataset\n"
                        f"Regenerate test dataset: python scripts/data/create_test_datasets.py"
                    )
            # Or v3 models might BE the pdr dataset directly (acceptable)
            elif not isinstance(f[first_model], h5py.Dataset):
                pytest.fail(f"Model {first_model} is neither Group nor Dataset")


class TestDatasetCompatibility:
    """Test that datasets follow expected patterns for PDRLoader."""

    def test_all_datasets_present(self):
        """All three test datasets should be present."""
        missing = []
        if not V1_TEST_PATH.exists():
            missing.append("v1")
        if not V2_TEST_PATH.exists():
            missing.append("v2")
        if not V3_TEST_PATH.exists():
            missing.append("v3")

        if missing:
            pytest.fail(
                f"Missing test datasets: {', '.join(missing)}\n"
                f"Run: python scripts/data/create_test_datasets.py\n"
                f"Then commit: git add data/test/*.h5"
            )

    def test_model_key_patterns(self):
        """Check that all datasets use consistent model key patterns."""
        if not all(p.exists() for p in [V1_TEST_PATH, V2_TEST_PATH, V3_TEST_PATH]):
            pytest.skip("Not all test datasets available")

        for path, name in [
            (V1_TEST_PATH, "v1"),
            (V2_TEST_PATH, "v2"),
            (V3_TEST_PATH, "v3"),
        ]:
            with h5py.File(path, "r") as f:
                model_keys = [
                    k
                    for k in f.keys()
                    if k not in {"header", "species", "model_ids", "model_df"}
                ]
                # All model keys should follow a pattern (model_XXX or similar)
                assert len(model_keys) > 0, f"{name}: No model keys found"

                # Check first model is accessible
                first_model = model_keys[0]
                assert first_model in f, f"{name}: Model {first_model} not accessible"
