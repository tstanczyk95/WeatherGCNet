import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class dataset_wind_nl(Dataset):
    def __init__(self, dataset_path, inputTimesteps, predictTimestep, train):
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        self.inputTimesteps = inputTimesteps
        self.predictTimestep = predictTimestep

        if train:
            x = data["train"]
        else:
            x = data["test"]

        # original order: T, V, C (vertex/city, timestep, channel/variable)
        # now the order is changed into: C, T, V
        self.x = torch.tensor(x).permute(2, 0, 1).double() 


    def __getitem__(self, item):
        x = self.x[:, item:item + self.inputTimesteps, :]
        y = self.x[0, item + self.inputTimesteps + self.predictTimestep - 1, :] # WIND SPEED of all the 7 cities (feature no. 0)

        return x, y

    def __len__(self):
        return self.x.shape[1] - self.inputTimesteps - self.predictTimestep + 1
