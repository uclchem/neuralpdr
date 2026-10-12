import abc
from collections.abc import Callable
from typing import ClassVar, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Scalar

from neuralpdr.data import SeriesBatch
from neuralpdr.model import SurrogateModel

# Reduces a per-device value over the devices along a mesh axis, e.g. jax.lax.pmean.
Reduction: TypeAlias = Callable[[Array, str], Array]


class Objective(eqx.Module):
    """Interface for what the training loop optimises.

    An objective computes the loss and metrics of a batch, the weights of its loss
    terms per epoch, and the validation loss. Its methods are called per device
    inside ``shard_map``, so they see that device's shard of the batch and return
    unreduced values. The training loop averages the losses over the devices and
    reduces every metric as declared in ``metric_reductions``.
    """

    # How the training loop reduces each metric over the devices.
    metric_reductions: ClassVar[dict[str, Reduction]]

    @abc.abstractmethod
    def __call__(
        self,
        model: SurrogateModel,
        batch: SeriesBatch,
        weights: Float[Array, " W"],
    ) -> tuple[Scalar, dict[str, Array]]:
        """Compute the training loss and metrics of a batch.

        Args:
            model (SurrogateModel): The model to train
            batch (SeriesBatch): One batch, or its shard on this device
            weights (Float[Array, " W"]): The weights of the loss terms, see ``weights``

        Returns:
            tuple[Scalar, dict[str, Array]]: The loss and the scalar metrics by name
        """

    @abc.abstractmethod
    def weights(self, epoch: int) -> Float[Array, " W"]:
        """Return the weights of the loss terms in the given epoch."""

    @abc.abstractmethod
    def validation_loss(self, model: SurrogateModel, batch: SeriesBatch) -> Scalar:
        """Compute the validation loss of a batch."""


class LatentODEObjective(Objective):
    """Loss of an ``EncoderEvolveDecoder`` on whole series.

    The training loss is a weighted sum of three terms: the decoded rollout against
    the data (``rollout_loss``), the rollout in latent space against the encoded
    data (``latent_loss``) and the decoded encoding against the data
    (``auto_loss``). The validation loss is the rollout loss alone, unweighted.
    """

    metric_reductions: ClassVar[dict[str, Reduction]] = {
        "nan_count": jax.lax.psum,
        "solver_steps_min": jax.lax.pmin,
        "solver_steps_med": jax.lax.pmean,
        "solver_steps_max": jax.lax.pmax,
        "rollout_loss": jax.lax.pmean,
        "latent_loss": jax.lax.pmean,
        "auto_loss": jax.lax.pmean,
    }

    def __call__(
        self,
        model: SurrogateModel,
        batch: SeriesBatch,
        weights: Float[Array, " 3"],
    ) -> tuple[Scalar, dict[str, Array]]:
        output = eqx.filter_vmap(model, in_axes=(0, 0, 0), out_axes=0)(
            batch.iv, batch.data, batch.aux
        )
        pred_y = output.pred
        evolved_z = output.extras["evolved_z"]
        auto_y = output.extras["auto_y"]
        direct_z = output.extras["direct_z"]
        steps = output.extras["steps"]
        # Catch the padded cells:
        valid_time_mask = batch.mask[:, :, None]
        # Catch the infinite values caused by max_step:
        valid_pred_mask = jnp.isfinite(pred_y[:, :, :1])
        # Obtain the loss for every sample in the batch after timestep 0.
        rollout_loss = jnp.mean(
            valid_time_mask * valid_pred_mask * (pred_y - batch.data) ** 2
        )
        latent_loss = jnp.mean(valid_time_mask * (direct_z - evolved_z) ** 2)
        auto_loss = jnp.mean(valid_time_mask * (auto_y - batch.data) ** 2)
        total_loss = (
            weights[0] * rollout_loss
            + weights[1] * latent_loss
            + weights[2] * auto_loss
        )
        return total_loss, {
            "nan_count": jnp.sum(~valid_pred_mask),
            "solver_steps_min": jnp.min(steps),
            "solver_steps_med": jnp.median(steps),
            "solver_steps_max": jnp.max(steps),
            "rollout_loss": rollout_loss,
            "latent_loss": latent_loss,
            "auto_loss": auto_loss,
        }

    def weights(self, epoch: int) -> Float[Array, " 3"]:
        """Weights of the rollout, latent and autoencoder loss: the rollout loss
        ramps up and the autoencoder loss down between epochs 15 and 30."""
        term0 = (
            [0.04 for i in range(15)]
            + np.linspace(0.04, 1.0, 15).tolist()
            + [1.0 for i in range(70)]
        )
        term1 = [1e-3 for i in range(100)]
        term2 = (
            [1.0 for i in range(15)]
            + np.linspace(1.0, 0.25, 15).tolist()
            + [0.25 for i in range(70)]
        )
        return jnp.array([term0, term1, term2]).T[epoch]

    def validation_loss(self, model: SurrogateModel, batch: SeriesBatch) -> Scalar:
        output = eqx.filter_vmap(model, in_axes=(0, 0, 0), out_axes=0)(
            batch.iv, batch.data, batch.aux
        )
        valid_mask = batch.mask[:, :, None]
        return jnp.mean(valid_mask * (output.pred - batch.data) ** 2)
