from pathlib import Path
import h5py
import pandas as pd
import numpy as np

DATA_PATH = "data/3d_pdr_dataset"
METADATA_PATH = "samples.csv"
STORE_PATH = "3dpdr_dataset_8192.h5"

if __name__=="__main__":
    store_path = Path(DATA_PATH)
    df = pd.read_csv(METADATA_PATH, index_col=0)
    with h5py.File(STORE_PATH, "a") as fh:
        fh.create_dataset("model_df", data=df.values, dtype="float32")            
        fh.create_dataset("model_ids", data=df.index.to_list(), dtype="S10" )
        for model_id, parameters in df.iterrows():
            print(model_id, end="\r")
            for datatype in ["pdr", "heat", "cool", "line", "opdp", "spop"]:
                pdr_data = np.genfromtxt(store_path / model_id / f"{model_id}.{datatype}.fin")
                fh.create_dataset(name=f"{model_id}/{datatype}", data=pdr_data, dtype="float32")
