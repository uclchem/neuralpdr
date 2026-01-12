import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# Mapping of version -> base config path and test dataset
TEST_DATASETS = {
    "v1": PROJECT_ROOT / "data" / "test" / "3dpdr_dataset_v1_test.h5",
    "v2": PROJECT_ROOT / "data" / "test" / "3dpdr_dataset_v2_test.h5",
    "v3": PROJECT_ROOT / "data" / "test" / "3dpdr_dataset_v3_test.h5",
}

BASE_CONFIGS = {
    "v1": PROJECT_ROOT / "configs" / "v1" / "ml4ps_paper" / "mlps_model_1.yaml",
    "v2": PROJECT_ROOT / "configs" / "v2" / "base.yaml",
    "v3": PROJECT_ROOT / "configs" / "v3" / "base.yaml",
}


def create_test_config(base_config_path: Path, tmp_dir: Path, version: str):
    """Load base YAML config, override values for fast testing and write test config file.

    Returns (config_dict, test_config_path).
    """
    # Load base YAML directly to mimic the file-based workflow
    with open(base_config_path, "r") as fh:
        config = yaml.safe_load(fh)

    # Override to use the small test dataset
    if version in TEST_DATASETS:
        config["dataset_path"] = str(TEST_DATASETS[version])

    # Minimal testing adjustments (mirror user edits to config file)
    config["save_file_path"] = str(tmp_dir)
    config["neptune_project"] = None
    config["neptune_tags"] = []
    config["plot_frequency"] = 999

    # Ensure only the first learning scheme, 1 epoch, constant lr
    if "learning_schemes" in config and config["learning_schemes"]:
        config["learning_schemes"] = [config["learning_schemes"][0]]
        config["learning_schemes"][0]["epochs"] = 1
        config["learning_schemes"][0]["lr_scheduler"] = "constant"
        config["learning_schemes"][0]["warmup_epochs"] = 0
    else:
        # Fallback: create a minimal learning scheme
        config["learning_schemes"] = [
            {
                "timeseries_fraction": 1.0,
                "epochs": 1,
                "learning_rate": 1e-3,
                "lr_scheduler": "constant",
                "warmup_epochs": 0,
            }
        ]

    # Small batch to ensure multiple batches per split
    config["batch_size"] = 8
    config["shuffle_every_n_epochs"] = 0

    # Save to temporary config file (this is what a user would run)
    test_config_path = tmp_dir / "test_config.yaml"
    with open(test_config_path, "w") as fh:
        yaml.dump(config, fh)

    return config, test_config_path


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
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
    assert Path(config["dataset_path"]).exists(), (
        f"Dataset not found: {config['dataset_path']}"
    )

    # Run the training script via subprocess to mimic user CLI
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "src" / "neuralpdr" / "train.py")),
        str(cfg_path),
    ]
    env = dict(**os.environ)
    # Force CPU to avoid GPU-related CI issues
    env["JAX_PLATFORM_NAME"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""

    # Remove any old cache files for this dataset to avoid cache/model index mismatches
    base = Path(config["dataset_path"]).with_suffix("")
    for p in base.parent.glob(f"{base.name}_*.pickle"):
        try:
            p.unlink()
        except Exception:
            pass

    try:
        subprocess.run(
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

    # Optional: at least one weights file (may not always exist depending on config)
    weight_files = list(out_dir.glob("weights_epoch_*.eqx"))

    list(weight_files)
    # no hard assert on weights - just log presence
    # assert len(weight_files) >= 0, "Weights file check (optional)"

    # If we made it here, the subprocess finished successfully and outputs were written
