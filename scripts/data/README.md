# Data Processing Scripts

This directory contains scripts to download and process the 3D-PDR datasets for use with the NeuralPDR package.

## Getting Started

**Step 1: Download the datasets from Zenodo**

The NeuralPDR project uses three datasets representing different cloud models:
- **v1** (8,192 1D uniform clouds): https://doi.org/10.5281/zenodo.13711173
- **v2** (variable density clouds, 4 metallicities): https://doi.org/10.5281/zenodo.7310832  
- **v3** (300K+ 3D GMC models): https://doi.org/10.5281/zenodo.15688233

Download these files and place them in the following structure:
```
data/zenodo/
├── v1/3pdr_dataset_8192.h5
├── v2/simulations.tgz
└── v3/3dpdr_dataset_v3.h5
```

**Step 2: Process all datasets**

Once downloaded, run the master workflow to process all three datasets into training-ready format:
```bash
./scripts/data/workflow_setup_all.sh
```

This will extract, convert, and add proper headers to all datasets, placing the final processed files in `data/processed/`. Processing takes 1-2 minutes (v1 and v2 are quick; v3 is an instant copy). You can optionally skip datasets with `--skip-v1`, `--skip-v2`, or `--skip-v3` flags. v3 smoothing can take up to one hour.

**Step 3: Use with NeuralPDR**
Continue in the scripts/train readme to see how to train with the datasets scripts/inference to use an
existing model for inference.

---

## Advanced Usage

### Processing Individual Datasets

If you only need specific datasets, you can process them individually:

```bash
# v1 only
python scripts/data/v1_add_headers.py

# v2 only (solar metallicity)
./scripts/data/workflow_v2.sh

# v3 only
./scripts/data/workflow_v3.sh
```

### v2: Working with Multiple Metallicities

By default, v2 processing uses only solar metallicity (Z1p0). For multi-metallicity studies:

```bash
# Add metallicity as a data column (useful for training with metallicity as an auxiliary parameter)
./scripts/data/workflow_v2.sh --all-metallicities --include-metallicity
```

**Available metallicities**: Z0p1 (0.1× solar), Z0p5 (0.5× solar), Z1p0 (1.0× solar, default), Z2p0 (2.0× solar)

When using `--all-metallicities`, model IDs are appended with `_Z0p1`, `_Z0p5`, `_Z1p0`, `_Z2p0` to distinguish them.

---

## Processing New Model Runs

If you generate new 3D-PDR model outputs and want to convert them to NeuralPDR format.

**Two-step process:**
1. **Convert raw outputs to HDF5**: `python scripts/data/v3_raw_to_h5.py <input_dir> output.h5`
2. **Add your custom headers**: `python scripts/data/vX_add_headers.py output.h5 final.h5`

---

