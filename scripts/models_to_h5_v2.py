from pathlib import Path
import h5py
import pandas as pd
import numpy as np

DATA_PATH = "../data/3d_pdr_dataset_v2/Z1p0"
STORE_PATH = "3dpdr_dataset_v2.h5"

SPECIES = ['H3+',
 'He+',
 'Mg',
 'H2+',
 'O2',
 'CH5+',
 'CH4+',
 'O+',
 'OH+',
 'Mg+',
 'C+',
 'CH4',
 'H2O+',
 'H3O+',
 'CO+',
 'O2+',
 'CH2',
 'H2O',
 'H+',
 'CH3+',
 'CH',
 'CH3',
 'HCO+',
 'CH2+',
 'C',
 'He',
 'CH+',
 'CO',
 'OH',
 'O',
 'H2',
 'H',
 'e-']

if __name__=="__main__":
    store_path = Path(DATA_PATH)
    model_ids = pd.DataFrame(sorted(Path(DATA_PATH).glob("*.params"))).map(lambda x: x.stem)
    params = [None] * len(model_ids)
    with h5py.File(STORE_PATH, "a") as fh:
        for idx, model_id in model_ids.iterrows():
            model_id = model_id.values[0]
            print(model_id, end="\r")
            params[idx] = np.genfromtxt(store_path / f"{model_id}.params")
            for datatype in ["pdr", "spop"]:
                pdr_data = np.genfromtxt(store_path / f"{model_id}.{datatype}.fin")
                fh.create_dataset(name=f"{model_id}/{datatype}", data=pdr_data, dtype="float32")
        fh.create_dataset("model_df", data=np.array(params), dtype="float32")    
        fh.create_dataset("model_ids", data=model_ids.values.flatten().tolist(), dtype="S10" )
        fh.create_dataset("species", data=SPECIES, dtype="S10" )

        
