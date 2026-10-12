"""Shared array type aliases (internal; not part of the public API).

Dimension names used throughout:
    B: batch size
    T: number of steps along the independent variable (time series length)
    F: number of data features
    A: number of auxiliary features
    L: number of latent features
    _: any size (not checked)

NumPy arrays are used on the host side while loading and normalising data;
JAX arrays are used from batching onwards.
"""

from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Array, Float, Int

# The series of one sample (one model) in the dataset, from loading up
# to batching. Covers the features (F columns), auxiliary features (A columns)
# and independent variable (1 column) alike. Each sample has its own length T,
# so these are not padded yet; ``pad_and_stack`` turns lists of them into the
# Batch* arrays below. They are NumPy arrays because the loader modifies them
# in place (NaN replacement, normalisation).
SampleSeries: TypeAlias = Float[np.ndarray, "T _"]

# Single sample, as seen by a model (i.e. inside vmap).
IV: TypeAlias = Float[Array, " T"]
Data: TypeAlias = Float[Array, "T F"]
Aux: TypeAlias = Float[Array, "T A"]
LatentSeries: TypeAlias = Float[Array, "T L"]
SolverSteps: TypeAlias = Int[Array, ""]

# Batched, padded samples, as produced by the data loader.
BatchIV: TypeAlias = Float[Array, "B T 1"]
BatchData: TypeAlias = Float[Array, "B T F"]
BatchAux: TypeAlias = Float[Array, "B T A"]

# The three groups of series every sample consists of: the independent variable,
# the (predicted) data features and the auxiliary (input-only) features.
SeriesGroup: TypeAlias = Literal["iv", "data", "aux"]
