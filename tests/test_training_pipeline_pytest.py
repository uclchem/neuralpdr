import math
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from neuralpdr.config import (
    DataMetadata,
    LearningScheme,
    read_as,
    read_conf,
    write_conf,
)

pytestmark = pytest.mark.integration


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Mapping of version -> base config path and test dataset
TEST_DATASETS = {
    "v1": PROJECT_ROOT / "data" / "test" / "3dpdr_dataset_v1_test.h5",
    "v2": PROJECT_ROOT / "data" / "test" / "3dpdr_dataset_v2_test.h5",
    "v3": PROJECT_ROOT / "data" / "test" / "3dpdr_dataset_v3_test.h5",
}

BASE_CONFIGS = {
    "v1": PROJECT_ROOT / "configs" / "v1" / "ml4ps_paper" / "mlps_model_1.toml",
    "v2": PROJECT_ROOT / "configs" / "v2" / "base.toml",
    "v3": PROJECT_ROOT / "configs" / "v3" / "base.toml",
}


def create_test_config(base_config_path: Path, tmp_dir: Path, version: str):
    """Load base TOML config, override values for fast testing and write test config file.

    Returns (config_dict, test_config_path).
    """
    config = read_conf(base_config_path)

    # Override to use the small test dataset
    if version in TEST_DATASETS:
        config = replace(config, dataset_path=TEST_DATASETS[version])

    # Ensure only the first learning scheme, 1 epoch, constant lr
    if not config.learning_schemes:
        ls = LearningScheme(lr_scheduler="constant", epochs=1, warmup_epochs=0)
    else:
        # Fallback: create a minimal learning scheme
        ls = LearningScheme(
            timeseries_fraction=1.0,
            epochs=1,
            learning_rate=1e-3,
            lr_scheduler="constant",
            warmup_epochs=0,
        )
    # Small batch to ensure multiple batches per split
    # Also shrink model architecture for fast compilation in tests
    config = replace(
        config,
        batch_size=8,
        shuffle_every_n_epochs=0,
        learning_schemes=[ls],
        enc_dec_width=32,
        width=32,
        bottleneck=8,
        save_file_path=tmp_dir,
    )

    # Save to temporary config file (this is what a user would run)
    test_config_path = tmp_dir / "test_config.toml"
    write_conf(config, test_config_path)

    return config, test_config_path


@pytest.mark.parametrize(
    "version",
    [
        "v1",
        pytest.param(
            "v2",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "v2 header has 39 names for 41 data columns, so 'visual_extinction' "
                    "reads a non-monotonic row index (see scripts/data/v2_add_headers.py)"
                ),
            ),
        ),
        "v3",
    ],
)
def test_training_pipeline_integration(version, tmp_path: Path):
    """Integration test that mirrors running `python src/neuralpdr/train.py config.yaml`.

    Uses the 128-sample test datasets and runs a single epoch of the first learning scheme.
    """
    base_config = BASE_CONFIGS.get(version)
    assert base_config is not None and base_config.exists(), (
        f"Base config for {version} not found: {base_config}"
    )

    # Prepare output dir
    out_dir = tmp_path / f"training_output_{version}"
    out_dir.mkdir()

    # Create and write test config (based on the real YAML)
    config, cfg_path = create_test_config(base_config, out_dir, version)

    # Sanity check the dataset exists
    assert config.dataset_path.exists(), f"Dataset not found: {config.dataset_path}"

    # Run the training script via subprocess to mimic user CLI
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "neuralpdr" / "train.py"),
        str(cfg_path),
    ]
    env = dict(**os.environ)
    # Force CPU to avoid GPU-related CI issues
    env["JAX_PLATFORM_NAME"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Skip plotting in integration tests
    env["NEURALPDR_PLOT_FREQ"] = "999999"

    # Remove any old cache files for this dataset to avoid cache/model index mismatches
    base = config.dataset_path.with_suffix("")
    for p in base.parent.glob(f"{base.name}_*.pickle"):
        try:
            p.unlink()
        except Exception:
            pass

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=env, timeout=600
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Training for {version} failed with return code {e.returncode}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"Training for {version} timed out: {e}")

    # Basic output assertions
    expected_files = ["hyperparameters.json", "data_metadata.json"]
    for ef in expected_files:
        assert (out_dir / ef).exists(), (
            f"Missing expected output file: {ef} in {out_dir}"
        )

    # Train, validation and test data come from different models.
    data_metadata = read_as(out_dir / "data_metadata.json", DataMetadata)
    train_ids = set(data_metadata.train_indices)
    val_ids = set(data_metadata.val_indices)
    test_ids = set(data_metadata.test_indices)
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)

    # Training must produce finite losses; a NaN validation loss means the run
    # silently diverged (e.g. a broken normalisation).
    epoch_losses = re.findall(r"train Loss: (\S+), val Loss: (\S+)", result.stdout)
    assert epoch_losses, f"No epoch losses found in stdout:\n{result.stdout}"
    for train_loss, val_loss in epoch_losses:
        assert math.isfinite(float(val_loss)), f"Non-finite val loss: {epoch_losses}"
        assert float(train_loss) > 0.0, f"Every training batch was NaN: {epoch_losses}"

    # Without a normalisation file, the computed one is saved and recorded for inference.
    if config.normalisations_file is None or not config.normalisations_file.exists():
        saved = out_dir / "normalisations.toml"
        assert saved.exists(), f"Computed normalisation not saved in {out_dir}"
        hyperparameters = read_conf(out_dir / "hyperparameters.json")
        assert hyperparameters.normalisations_file == saved

    # Optional: at least one weights file (may not always exist depending on config)
    weight_files = list(out_dir.glob("weights_epoch_*.eqx"))

    list(weight_files)
    # no hard assert on weights - just log presence
    # assert len(weight_files) >= 0, "Weights file check (optional)"

    # If we made it here, the subprocess finished successfully and outputs were written
