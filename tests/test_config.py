from pydantic import ValidationError
import pytest

from neuralpdr.config import FNO, Latent, LearningScheduler, LearningScheme, Split
from neuralpdr.config import tomllib


@pytest.mark.parametrize("train,validate,test", [(0.7, 0.2, 0.1)])
def test_cross_validation_split(train: float, validate: float, test: float):
    Split(train, validate, test)


@pytest.mark.parametrize(
    "train,validate,test,exc",
    [(0.3, 0.3, 0.3, ValueError), (0.4, 0.4, 0.4, ValueError)],
)
def test_cross_validation_split_err(
    train: float, validate: float, test: float, exc: Exception
):
    data = {"train": train, "validate": validate, "test": test}
    with pytest.raises(exc, match=f"{data}"):
        Split(train, validate, test)


@pytest.mark.parametrize("scheduler", ["sgdr", "constant"])
def test_learning_scheme(scheduler: LearningScheduler):
    LearningScheme(scheduler, 2, 0.2)


def test_learning_scheme_err():
    with pytest.raises(ValidationError):
        LearningScheme("unreal", 42, 0.314)
