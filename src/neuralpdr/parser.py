import argparse
import logging
import os
import warnings
from pathlib import Path
from string import Template

from neuralpdr.config import tomllib

logger = logging.getLogger(__name__)


def find_project_root(start_path):
    """Find the project root by looking for pyproject.toml or .git directory.

    Args:
        start_path: Path to start searching from

    Returns:
        Path to project root, or current working directory if not found
    """
    current = Path(start_path).resolve()

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        # Check for project markers
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    # Fall back to current working directory
    return Path.cwd()


def resolve_path(path_str, config_file_path, path_type="config"):
    """Resolve a path string to an absolute Path object.

    Args:
        path_str: Path string to resolve
        config_file_path: Path to the config file (for context)
        path_type: Type of path ("config", "data", "output") for different resolution rules

    Returns:
        Absolute Path object

    Rules:
    1. Config dependency paths (input_features, normalisations):
       - Relative paths resolve to project root
    2. Dataset and output paths:
       - Environment variable substitution: ${VAR_NAME}/path
       - Absolute paths used as-is
       - Relative paths resolve to project root (with warning for data paths)
    3. Home directory expansion: ~/path
    """
    if path_str is None:
        return None

    # Convert to string if Path object
    path_str = str(path_str)

    # Environment variable substitution
    if "${" in path_str:
        # Use string.Template for safe substitution
        try:
            template = Template(path_str)
            path_str = template.substitute(os.environ)
        except KeyError as e:
            raise ValueError(
                f"Environment variable {e} not found in path: {path_str}\n"
                f"Please set the environment variable or use an absolute path."
            )

    # Convert to Path object
    path = Path(path_str).expanduser()

    # If already absolute, return it
    if path.is_absolute():
        return path

    # Resolve relative paths relative to project root
    project_root = find_project_root(config_file_path)
    resolved_path = (project_root / path).resolve()

    # Log info for relative data/output paths (config paths are expected to be relative)
    if path_type in ["data", "output"]:
        logger.info(
            f"Relative {path_type} path '{path_str}' resolved to '{resolved_path}'. "
            f"Consider using absolute paths or environment variables for {path_type} paths."
        )

    return resolved_path


def validate_path(path, path_name, require_exists=False):
    """Validate a path and optionally check if it exists.

    Args:
        path: Path object to validate
        path_name: Name of the path for error messages
        require_exists: If True, raise error if path doesn't exist; if False, just warn
    """
    if path is None:
        return

    if not path.exists():
        message = f"{path_name}: '{path}' does not exist"
        if require_exists:
            raise FileNotFoundError(message)
        else:
            warnings.warn(message, UserWarning)


def resolve_config_paths(config, config_file_path):
    """Resolve all paths in config to absolute paths.

    Args:
        config: Configuration dictionary
        config_file_path: Path to the config file

    Returns:
        Config with all paths resolved to absolute Path objects (as strings)
    """
    # Define which fields are which type of path
    config_paths = [
        "input_features_file",
        "normalisations_file",
        "checkpoint_file",
    ]

    data_paths = [
        "dataset_path",
    ]

    output_paths = [
        "save_file_path",
    ]

    # Resolve config dependency paths (relative to config file)
    for field in config_paths:
        if field in config and config[field]:
            resolved = resolve_path(config[field], config_file_path, path_type="config")
            # Validate input files should exist
            if field in ["input_features_file", "normalisations_file"]:
                validate_path(resolved, field, require_exists=False)
            config[field] = str(resolved)

    # Resolve data paths (can use env vars, warn if relative)
    for field in data_paths:
        if field in config and config[field]:
            resolved = resolve_path(config[field], config_file_path, path_type="data")
            # Don't require dataset to exist (might be on cluster)
            config[field] = str(resolved)

    # Resolve output paths (can use env vars, warn if relative)
    for field in output_paths:
        if field in config and config[field]:
            resolved = resolve_path(config[field], config_file_path, path_type="output")
            # Output paths don't need to exist yet
            config[field] = str(resolved)

    return config
