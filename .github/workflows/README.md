# GitHub Actions Workflows

This directory contains automated CI/CD workflows for the neuralpdr project.

## Workflows

### 1. `test.yml` - Continuous Integration Tests

**Triggers:** Push to main/develop, Pull Requests, Manual

This workflow runs automatically on every push and pull request. It:
- Tests on multiple OS (Ubuntu, macOS)
- Tests on multiple Python versions (3.10, 3.11, 3.12)
- Runs all unit tests
- **Runs integration tests** using minimal test datasets (~6 MB total)

**Minimal Test Datasets:**
The repository includes small test datasets in `data/test/` that allow integration tests to run in CI:
- `3dpdr_dataset_v1_test.h5` - 10 models, 4 MB
- `3dpdr_dataset_v2_test.h5` - 10 models, 1.3 MB
- `3dpdr_dataset_v3_test.h5` - 10 models, 0.26 MB

These are automatically used by the tests when available, allowing full integration testing in CI without requiring the multi-GB full datasets.

### 2. `integration-test.yml` - Full Integration Tests

**Triggers:** Manual only

This workflow is for running full integration tests with actual datasets. Use this when:
- You have datasets available via a download URL
- You want to test with real data before a release

**How to run:**
1. Go to the "Actions" tab on GitHub
2. Select "Integration Tests (with datasets)"
3. Click "Run workflow"
4. (Optional) Provide a URL to a tar.gz file containing the datasets
5. Click "Run workflow"

**Dataset preparation for full tests:**

If you want to run integration tests in CI, you can:

1. **Upload datasets to GitHub Release:**
   ```bash
   cd data/processed
   tar -czf neuralpdr-datasets.tar.gz *.h5
   # Upload to GitHub Release
   ```

2. **Use the release URL in the workflow:**
   - URL format: `https://github.com/uclchem/neuralpdr/releases/download/v1.0.0/neuralpdr-datasets.tar.gz`

3. **Or store as GitHub Actions artifact** (for temporary testing)

## Local Testing

To run tests locally (same as CI does):

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_data_loaders.py -v
python -m pytest tests/test_pdrloader_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=src/neuralpdr --cov-report=html
```

## Test Coverage

- **Unit Tests** (`test_data_loaders.py`): Test HDF5 structure validation
- **Integration Tests** (`test_pdrloader_integration.py`): Test PDRLoader with real datasets
  - Use minimal datasets (10 models each) for CI
  - Automatically fall back to full datasets if available locally

## Generating Minimal Test Datasets

Minimal test datasets are already included in the repository. To regenerate them:

```bash
python scripts/data/create_test_datasets.py
```

This will create 10-model subsets of each dataset in `data/test/` (~6 MB total).

## Adding More Tests

When adding new tests:
1. Place test files in `tests/` directory
2. Use `pytest.skip()` for tests requiring large datasets:
   ```python
   if not dataset_path.exists():
       pytest.skip(f"Dataset not found at {dataset_path}")
   ```
3. Tests will automatically run in CI on next push

## Troubleshooting

### Tests fail in CI but pass locally
- Check Python version compatibility
- Verify all dependencies are in `requirements.txt`
- Check for OS-specific issues (file paths, etc.)

### Integration tests always skip
- Expected behavior in CI (datasets not available)
- Run locally with datasets or use `integration-test.yml` workflow

### JAX installation issues
- JAX is a required dependency
- If tests fail with JAX import errors, check `requirements.txt`
