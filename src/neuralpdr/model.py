import math
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve


def get_model(n_input_features:int, width:int, depth:int, model_key:jax.random.PRNGKey, activation=jax.nn.softplus) -> eqx.Module:
    """Create a neural network model for the neural ODE.

    Args:
        n_input_features (int): The number of input features
        width (int): The width of the neural network
        depth (int): The depth of the neural network
        model_key (jax.random.PRNGKey): The random key to use for the initialization

    Returns:
        eqx.Module: The neural network model
    """
    return eqx.nn.MLP(
        in_size=n_input_features,
        out_size=n_input_features,
        width_size=width,
        depth=depth,
        key=model_key,
        activation=activation,
    )

@eqx.filter_jit
def solve_ODE(model: eqx.Module, avs: jax.Array, y0: jax.Array) -> jax.Array:
    """Solve the ODE for a given model and set of visual extinctions.

    Args:
        model (eqx.Module): The NN part of the neuralOE
        avs (jax.Array): The visual extinctions to solve for
        y0 (jax.Array): The initial conditions for the ODEs

    Returns:
        jax.Array: _description_
    """
    solution = diffeqsolve(
        ODETerm(lambda av, y, args: model(y)),
        Tsit5(),
        t0=avs[0],
        t1=avs[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
        saveat=SaveAt(ts=avs),
        max_steps=10000, # Increase maximum number of steps
    )
    return solution.ys

class trunc_init:
    def __init__(self, scale: float, lower: float, upper: float):
        """Truncated normal initialization for the weights of the neural network.

        Args:
            scale (float): scale factor within the truncated normal distribution
            lower (float): cutoff value for the lower bound of the truncated normal distribution
            upper (float): cutoff value for the upper bound of the truncated normal distribution
        """
        self.scale = scale
        self.lower = lower
        self.upper = upper

    def __call__(self, weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        """Initialize the weights of the neural network with a truncated normal distribution.

        Args:
            weight (jax.Array): The weight matrix to initialize
            key (jax.random.PRNGKey): The random key to use for the initialization

        Returns:
            jax.Array: The initialized weight matrix
        """
        out, in_ = weight.shape
        stddev = math.sqrt(self.scale / in_)
        return stddev * jax.random.truncated_normal(
            key, shape=(out, in_), lower=self.lower, upper=self.upper
        )

    def __repr__(self):
        return f"trunc_init(scale={self.scale}, lower={self.lower}, upper={self.upper})"


def is_linear(layer: eqx.Module) -> bool:
    """Check if a function is a linear layer.

    Args:
        layer (eqx.Module):  The layer to check

    Returns:
        bool: True if the layer is a linear layer, False otherwise
    """
    return isinstance(layer, eqx.nn.Linear)


def get_weights(mlp: eqx.Module) -> list[jax.Array]:
    """Get the weights of the neural network

    Args:
        mlp (eqx.Module): The neural network

    Returns:
        list[jax.Array]: List of the weights of the neural network
    """
    return [
        x.weight
        for x in jax.tree_util.tree_leaves(mlp, is_leaf=is_linear)
        if is_linear(x)
    ]


def get_norm(mlp: eqx.Module, order:int=1) -> list[jax.Array]:
    """Get the norms of the weights of the neural network

    Args:
        mlp (eqx.Module): The neural network
        order (int, optional): The order of the norm. Defaults to 1.

    Returns:
        list[jax.Array]: List of the norms of the weights of the neural network
    """
    return [jnp.linalg.norm(w, ord=order) for w in get_weights(mlp)]


def init_linear_weight(
    model: eqx.Module,
    init_fn: Callable[[jax.Array, jax.random.PRNGKey], jax.Array],
    key: jax.random.PRNGKey,
) -> eqx.Module:
    """Initialize the weights of the neural network.

    Args:
        model (eqx.Module): The neural network
        init_fn (callable): Function to initialize the weights with signature (weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array
        key (jax.random.PRNGKey): The random key to use for the initialization

    Returns:
        eqx.Module: The neural network with newly initialized weights
    """
    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model