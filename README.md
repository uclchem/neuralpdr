# NeuralPDR

If you use this code, please cite: https://doi.org/10.1088/2632-2153/ade4ee

In this project we emulate the Photodissociation Region (PDR) code 3D-PDR for 1D-dimensional uniform clouds (v1), clouds of varying density (v2) and a 3D model of a giant molecular cloud (v3)
3D-PDR code solves the cooling, heating and chemistry as we move along lines of sights into a cloud (changing Av). 

We present here Augmented Neural Ordinary Differential Equations that act as surrogate models / emulators for
this chemistry. Taking densities (constant), cosmic ray ionisations (constant), visual extinctions, radiation field
as additional parameters and the chemical abundances and temperatures as normal features. 

# Usage
After downloading the datasets (listed below), the code can be trained using `python src/neuralpdr/train.py CONFIGURATION_YAML`, after training, you can refer the results using `python src/neuralpdr/inference.py  --dataset_path $DATASET_PATH --model_dir $MODEL_DIRECTORY --weights_file WEIGHTS_TO_USE`

In order to use the Neptune callback, please provide your neptune key as `"NEPTUNE_API_TOKEN= ...` in secret_api_key.py.

# Datasets
The first dataset can be found on Zenodo: https://doi.org/10.5281/zenodo.13711173
The second dataset can be retrieved from: https://doi.org/10.5281/zenodo.7310832
The third dataset can be found at: https://doi.org/10.5281/zenodo.15688233
