# %%

import matplotlib.pyplot as plt
import numpy as np

from train import DataLoader, h5py_load, shuffle_and_split

if __name__ == "__main__":
    dataset_path = "../3dpdr_dataset_8192.h5"    
    batch_size = 16
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    input_features = [
        "visual_extinction",
        "tgas",
        "tdust",
        "density",
        "radfield",
        "zeta_init",
        "H",
        "H2",
        "C",
        "C2",
        "CO",
        "e-",
    ]
    av_eps = 1e-11
    data_eps = [1e-10,] + [1e-20,]*11
    index_range = (0, 300)

    model_df = h5py_load(
        dataset_path, "model_df", return_dataframe=True, columns=["g_uv", "n_H", "zeta"]
    )
    model_df.index = h5py_load(dataset_path, "model_ids", text=True)
    model_df.columns = ["radfield_init", "density_init", "zeta_init"]

    train_indices, val_indices, test_indices = shuffle_and_split(
        model_df, train_split=train_split, val_split=val_split, test_split=test_split
    )

    # Loading data
    train_dataloader = DataLoader(
        dataset_path,
        model_df,
        index_range=index_range,
        model_indices=train_indices.index,
        input_features=input_features,
        batch_size=batch_size,
        stage="train",
        normalize_kwargs = dict(eps=eps)
    )
    
    datas = np.vstack(list(train_dataloader.raw_data.values()))
    normalized_data = np.vstack(list(train_dataloader.data.values()))
    fig, ax = plt.subplots(datas.shape[-1], 2, figsize=(40, 20))
    for i in range(datas.shape[1]):
        ax[i][0].hist(datas[:, i], bins=25, label=input_features[i])
        ax[i][1].hist(normalized_data[:, i], bins=25, label=input_features[i])
    plt.savefig("comparison.png")

# %%
normalized_data.shape
# %%
normalized_data[:50]
# %%
fig, ax = plt.subplots(datas.shape[-1], 2, figsize=(40, 20))
labels = list(train_dataloader.col_index_by_header_to_load.values()) + ["zeta_init",]
for i in range(datas.shape[1]):
    ax[i][0].hist(np.log10(datas[:, i] + 1e-20), bins=25, label=input_features[i])
    ax[i][0].set_ylabel(labels[i])
    ax[i][1].hist(normalized_data[:, i], bins=25, label=input_features[i])
plt.savefig("comparison.png")
# %%
import pandas as pd

pd.DataFrame(normalized_data, columns=list(train_dataloader.col_index_by_header_to_load.values()) + ["zeta_init",]).describe()
# %%
