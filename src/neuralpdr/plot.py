import logging

import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

# from model import solve_ODE


def plot_batch(mlp, loader, epoch=None, save_file_path=".", n_samples=16):
    av, data, aux = loader[np.random.randint(0, len(loader))]
    pred, z, ae_y, ae_z, _ = eqx.filter_vmap(mlp, in_axes=(0, 0, 0))(
        av[:, :, 0], data, aux
    )
    av = loader.inv_normalize(av, "iv")
    data = loader.inv_normalize(data, "data")
    aux = loader.inv_normalize(aux, "aux")
    pred = loader.inv_normalize(pred, "data")
    ae_y = loader.inv_normalize(ae_y, "data")

    names = loader.model_indices
    if len(pred) > n_samples:
        names = names[:n_samples]
        av = av[:n_samples]
        data = data[:n_samples]
        pred = pred[:n_samples]
        ae_y = ae_y[:n_samples]
        aux = aux[:n_samples]
        z = z[:n_samples]
        ae_z = ae_z[:n_samples]
    # print("Plotting with shapes:", av.shape, data.shape, pred.shape, aux.shape, ae_y.shape, z.shape, ae_z.shape)
    n_features = len(loader.data_features)
    fig, axes = plt.subplots(n_features // 2 + 4, 2, figsize=(11, 2.1 * n_features))
    axes = axes.flatten()
    ax_count = 0
    for ax in axes:
        if ax_count < n_features:
            ax.plot(
                av[:, :, 0].T,
                data[:, :, ax_count].T,
                linestyle="solid",
                label=names if ax_count == 0 else None,
            )
            ax.set_prop_cycle(None)
            ax.plot(
                av[:, :, 0].T,
                ae_y[:, :, ax_count].T,
                linestyle="dotted",
                # label="autoregress" if ax_count == 0 else None,
            )
            ax.set_prop_cycle(None)
            ax.plot(
                av[:, :, 0].T,
                pred[:, :, ax_count].T,
                linestyle="dashed",
                # label="rollout" if ax_count == 0 else None,
            )
            ax.set_ylabel(loader.data_features[ax_count])
        elif ax_count < n_features + aux.shape[-1]:
            ax.plot(
                av[:, :, 0].T,
                aux[:, :, ax_count % n_features].T,
                linestyle="solid",
            )
        else:
            ax.plot(av[0, :, 0].reshape(-1, 1), z[0, :, :])
            ax.plot(av[0, :, 0].reshape(-1, 1), ae_z[0, :, :], linestyle="solid")
            break
        ax_count += 1
    fig.legend(ncol=3, loc="upper center")
    savepath = save_file_path / f"batch_{epoch}.png"
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath)
    plt.close(fig)
