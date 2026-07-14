import math
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,
)


class LatentMLP(eqx.Module):
    mlp: eqx.Module

    # scale: jnp.ndarray
    def __init__(
        self,
        n_input_features: int,
        width: int,
        depth: int,
        key: jax.Array,
        activation: Callable,
        final_activation: Callable | None = None,
        n_output_features: int | None = None,
        **kwargs,
    ) -> None:
        """Create a neural network model for the neural ODE.

        Args:
            n_input_features (int): The number of input features
            width (int): The width of the neural network
            depth (int): The depth of the neural network
            model_key (jax.random.PRNGKey): The random key to use for the initialization

        Returns:
            eqx.Module: The neural network model
        """
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=n_input_features,
            out_size=n_output_features if n_output_features else n_input_features,
            width_size=width,
            depth=depth,
            key=key,
            activation=activation,
            final_activation=final_activation if final_activation else lambda x: x,
        )
        # self.scale = jnp.array(0.001)

    def __call__(self, t, y, args):
        z = jnp.concatenate([y, args], axis=-1)
        return self.mlp(z)


def get_flexible_block(
    input_size: int,
    output_size: int,
    layers: list[int],
    key: jax.Array,
    activation,
    final_activation=None,
):
    layer_keys = jax.random.split(key, len(layers) + 2)
    jax_layers = [
        eqx.nn.Linear(input_size, layers[0], key=layer_keys[0]),
        eqx.nn.Lambda(activation),
    ]
    for idx, (s1, s2) in enumerate(zip(layers[:-1], layers[1:])):
        jax_layers.append(
            eqx.nn.Linear(s1, s2, key=layer_keys[idx + 2])
        )  # start after two first keys
        jax_layers.append(eqx.nn.Lambda(activation))
    jax_layers.append(eqx.nn.Linear(s2, output_size, key=layer_keys[1]))
    if final_activation:
        jax_layers.append(eqx.nn.Lambda(final_activation))
    return eqx.nn.Sequential(jax_layers)


@eqx.filter_jit
def latent_solve_factory(model, initial_y, ivs, args):
    # Take out the arrays from the static part here

    @jax.jit
    def rollout_step(carry, iv_idx):
        y, ys, ivs, args, model_arrays = carry
        model = eqx.combine(model_arrays, model_static)
        solution = diffeqsolve(
            ODETerm(lambda t, y, args: model(t, y, args)),
            Tsit5(),
            t0=ivs[iv_idx],
            t1=ivs[iv_idx + 1],
            dt0=None,
            y0=y,
            stepsize_controller=PIDController(
                rtol=1e-3, atol=1e-6, pcoeff=0.4, icoeff=0.3
            ),
            saveat=SaveAt(
                t1=True,
            ),
            args=args[iv_idx],
            throw=False,
        )
        y = solution.ys[-1]
        ys = ys.at[iv_idx].set(y)  # Use the `.at[index].set(value)` operation to assign
        model_arrays, _ = eqx.partition(model, eqx.is_array)
        return (
            y,
            ys,
            ivs,
            args,
            model_arrays,
        ), y  # Return updated state and the value for next step

    n_steps = len(ivs) - 1  # The number of iterations
    ys = jnp.zeros(
        (n_steps, initial_y.shape[0])
    )  # Preallocate `ys` with the correct size
    model_arrays, model_static = eqx.partition(model, eqx.is_array)

    carry_init = (
        initial_y,
        ys,
        ivs,
        args,
        model_arrays,
    )
    _, scan_out = jax.lax.scan(rollout_step, carry_init, jnp.arange(n_steps))
    return scan_out


# def mlp_with_aux_factory(mlp, ts, args):
#     class mlp_with_aux(eqx.Module):
#         mlp: eqx.Module
#         ts: jax.Array
#         args: jax.Array
#         def __init__(self, mlp, ts, args):
#             self.mlp = mlp
#             self.ts = ts
#             self.args = args

#         def __call__(self, t, x):
#             return self.mlp(t, x, )
#     return mlp_with_aux(mlp, ts, args)


class EncoderEvolveDecoder(eqx.Module):
    latent_mlp: eqx.Module
    enc: eqx.Module
    dec: eqx.Module

    def __init__(
        self,
        input_features,
        enc_dec_width,
        enc_dec_depth,
        latent_width,
        latent_depth,
        latent_weight_scale,
        latent_weight_truncation,
        latent_bottleneck: int = 4,
        n_aux_features: int = False,
        *,
        latent_final_activation: Callable,
        keys,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_mlp = LatentMLP(
            latent_bottleneck + n_aux_features,
            latent_width,
            latent_depth,
            key=keys[0],
            activation=jax.nn.softplus,
            final_activation=latent_final_activation,
            n_output_features=latent_bottleneck,
        )
        weight_initializer = trunc_init(
            latent_weight_scale, -latent_weight_truncation, latent_weight_truncation
        )
        self.latent_mlp = init_linear_weight(
            self.latent_mlp, weight_initializer, keys[0]
        )
        self.enc = get_flexible_block(
            len(input_features),
            latent_bottleneck,
            layers=[
                enc_dec_width,
            ]
            * enc_dec_depth,
            key=keys[1],
            activation=jax.nn.softplus,
        )
        self.dec = get_flexible_block(
            latent_bottleneck,
            len(input_features),
            layers=[
                enc_dec_width,
            ]
            * enc_dec_depth,
            key=keys[2],
            activation=jax.nn.softplus,
        )

    def __call__(
        self, ivs: jax.Array, y: jax.Array, aux: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        # encode
        z = jax.vmap(self.enc, in_axes=(0,))(y)
        initial_z = z[0]
        # evolve
        # latent_mlp_with_aux = mlp_with_aux_factory(self.latent_mlp, ivs, aux)
        solution = diffeqsolve(
            ODETerm(
                lambda t, y, args: self.latent_mlp(
                    t,
                    y,
                    aux[
                        jnp.searchsorted(ivs, t, method="scan_unrolled", side="right")
                        - 1
                    ],
                )
            ),
            Tsit5(),
            t0=ivs[0],
            t1=ivs[-1],
            dt0=None,
            y0=initial_z,
            stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
            saveat=SaveAt(ts=ivs),
            throw=False,
            max_steps=16384,
        )
        # Return the decoded rollout, the rollout in latent space, the decoding of the latent space and the latent space
        return (
            jax.vmap(self.dec, in_axes=(0,))(solution.ys),
            solution.ys,
            jax.vmap(self.dec, in_axes=(0,))(z),
            z,
            solution.stats["num_steps"],
        )


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

    def __call__(self, weight: jax.Array, key: jax.Array) -> jax.Array:
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


def get_norm(mlp: eqx.Module, order: int = 1) -> list[jax.Array]:
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
    init_fn: Callable[[jax.Array, jax.Array], jax.Array],
    key: jax.Array,
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
