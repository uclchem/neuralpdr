import chex
import jax.numpy as jnp
import optax
from typing import Sequence
import subprocess


def get_git_info() -> dict[str, str]:
    """Get the git info of the current repository

    Returns:
        dict[str, str]: Dict with keys 'git_commit_hash', 'git_info' and 'git_diff_to_HEAD' with the corresponding values
    """
    git_info = {}
    try:
        git_info["git_commit_hash"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
        git_info["git_info"] = (
            subprocess.check_output(["git", "log", "-1"]).decode().strip()
        )
        git_info["git_diff_to_HEAD"] = (
            subprocess.check_output(["git", "diff", "HEAD"]).decode().strip()
        )
    except Exception as e:
        print(f"Error getting git info: {e}")
    return git_info


def join_schedules(
    schedules: Sequence[optax.Schedule], boundaries: Sequence[int]
) -> optax.Schedule:
    """Sequentially apply multiple schedules.

    Args:
      schedules: A list of callables (expected to be optax schedules). Each
        schedule will receive a step count indicating the number of steps since
        the previous boundary transition.
      boundaries: A list of integers (of length one less than schedules) that
        indicate when to transition between schedules.

    Returns:
      schedule: A function that maps step counts to values.
    """

    def schedule(step: chex.Numeric) -> chex.Numeric:
        output = schedules[0](step)
        for boundary, schedule in zip(boundaries, schedules[1:]):
            output = jnp.where(step < boundary, output, schedule(step - boundary))
        return output

    return schedule
