# emulating-3dpdr

In this project we try to emulate the Photodissociation Region (PDR) code 3D-PDR for 1D-dimensional uniform clouds, 
the code solves the cooling, heating and chemistry as we move into a cloud (changing Av). 

We present here Augmented Neural Ordinary Differential Equations that act as surrogate models / emulators for
this chemistry. Taking densities (constant), cosmic ray ionisations (constant), visual extinctions, radiation field
as additional parameters and the chemical abundances and temperatures as normal features. 

In order to use the Neptune callback, please provide your neptune key as `"NEPTUNE_API_TOKEN= ...` in secret_api_key.py.

The experiments in the article can be reproduced by downloading the data from https://zenodo.org/doi/10.5281/zenodo.13711173, placing it in the root
directory and then running the script run_mlps_models.sh and running inference with `run_inference.sh`.



