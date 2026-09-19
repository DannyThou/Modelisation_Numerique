import torch
from torch.utils.data import Dataset
import pandas as pd

class ExoplanetDataset(Dataset):
    def __init__(self, 
                 data_dir,
                 ext=".csv",
                 transform=None):
        
        """
        Args:
            data_dir (str): Fichier csv avec les données d'exoplanètes.
            ext (str): Extension du fichier csv.
            transform (callable, optional): Transformations à appliquer aux données. Par défaut None.
        """


        self.data_dir = data_dir
        self.ext = ext
        self.transform = transform

        data = pd.read_csv(self.data_dir + self.ext)
        labels_bruts = data["LABEL"].values


        if set(labels_bruts) == {0, 1}:
            self.labels = labels_bruts.astype("int64")
        

        elif set(labels_bruts) == {1, 2}:
            self.labels = (labels_bruts - 1).astype("int64")
        

        else:
            raise ValueError(
                f"Labels inattendus trouvés dans LABEL : {set(labels_bruts)}. "
                "On attend soit {0,1}, soit {1,2}."
            )

        flux_cols = [col for col in data.columns if col.startswith("FLUX")]
        self.data_flux = data[flux_cols].values.astype("float32")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        flux = self.data_flux[idx]
        label = self.labels[idx]

        flux = torch.tensor(flux).unsqueeze(0)
        label = torch.tensor(label)

        if self.transform:
            flux = self.transform(flux)
        
        return flux, label
    
Exo = ["Sans exoplanète", 
       "Avec exoplanète"]

labels = [0, 1]

CDICT = dict(zip(labels, Exo))

