# NeuralPDR

If you use this code, please cite: https://doi.org/10.1088/2632-2153/ade4ee

In this project we emulate the Photodissociation Region (PDR) code 3D-PDR for 1D-dimensional uniform clouds (v1), clouds of varying density (v2) and a 3D model of a giant molecular cloud (v3)
3D-PDR code solves the cooling, heating and chemistry as we move along lines of sights into a cloud (changing Av). 

We present here Augmented Neural Ordinary Differential Equations that act as surrogate models / emulators for
this chemistry. Taking densities (constant), cosmic ray ionisations (constant), visual extinctions, radiation field
as additional parameters and the chemical abundances and temperatures as normal features. 

# Usage

## Dataset Preparation
Download the datasets from Zenodo (links below) and place them in `data/zenodo/` following this structure:
```
data/zenodo/
├── v1/3pdr_dataset_8192.h5
├── v2/simulations.tgz
└── v3/3dpdr_dataset_v3.h5
```

Process the raw datasets into training-ready format by running the header processing scripts:
```bash
# Process all datasets
./scripts/data/workflow_setup_all.sh

# Or process individually
python scripts/data/v1_add_headers.py   # for v1
./scripts/data/workflow_v2.sh           # for v2
./scripts/data/workflow_v3.sh           # for v3 
```

This creates processed HDF5 files in `data/processed/`:
```
data/processed/
├── 3dpdr_dataset_v1.h5
├── 3dpdr_dataset_v2.h5
└── 3dpdr_dataset_v3.h5
```

The datasets correspond to different cloud models: v1 (uniform 1D clouds), v2 (varying density clouds), v3 (3D giant molecular cloud).

Alternatively, use the test datasets in `data/test/` for quick validation without downloading the full Zenodo datasets.

## Training
Train the model using a configuration file from `configs/`. Example configurations are provided for each dataset version in `configs/v1/`, `configs/v2/`, and `configs/v3/`.

```bash
python src/neuralpdr/train.py configs/v2/base.yaml
```

Model checkpoints and logs are saved to the directory specified in the configuration file.

## Inference
Run inference on a trained model by specifying the dataset, model directory, and weights file:

```bash
python src/neuralpdr/inference.py --dataset_path data/processed/3dpdr_dataset_v2.h5 --model_dir runs/model_name --weights_file best_weights.pkl
```

## Neptune Callback (Deprecated)
**Note:** Neptune will be deprecated as of March 2026, so its call will stop functioning and soon be removed from the codebase.

# Datasets
The first dataset can be found on Zenodo: https://doi.org/10.5281/zenodo.13711173
The second dataset can be retrieved from: https://doi.org/10.5281/zenodo.7310832
The third dataset can be found at: https://doi.org/10.5281/zenodo.15688233
