import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from model import solve_ODE


def plot_batch(mlp, loader, epoch=None, save_file_path=".", n_samples=16):
        av, data = loader[0]
        pred = eqx.filter_vmap(solve_ODE, in_axes=(None, 0, 0))(
            mlp, av, data[:, 0, :]
        )
        av = loader.inv_normalize_av(av)
        data = loader.inv_normalize_data(data)
        pred = loader.inv_normalize_data(pred)
        names = loader.model_df.index[:n_samples]
        av = av[:n_samples]
        data = data[:n_samples]
        pred = pred[:n_samples]
        n_features = loader.n_features
        fig, ax = plt.subplots(n_features, 1, figsize=(7, 2.5*n_features))
        for feature in range(0, n_features):
            ax[feature].plot(
                av.T,
                data[:, :, feature].T,
                linestyle="solid",
                label=names if feature == 0 else None,
            )
            ax[feature].set_prop_cycle(None)
            ax[feature].plot(
                av.T,
                pred[:, :, feature].T,
                linestyle="dashed",
            )
            ax[feature].set_ylabel(loader.loaded_columns[feature])
        fig.legend(ncol=3, loc="upper center")
        savepath = save_file_path / f"batch_{epoch}.png"
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath)

