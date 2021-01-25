import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.io import loadmat
import numpy as np


def get_train_valid_loader(dataset_path):
    mat = loadmat(dataset_path)

    x_mat = mat["Xtr"]
    y_mat = mat["Ytr"]
 
    x = torch.tensor(x_mat).permute(0, 3, 2, 1).double()
    y = torch.tensor(y_mat).double()

    train_val_dataset = TensorDataset(x, y)

    num_train = len(train_val_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    np.random.seed(123)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_val_dataset, batch_size=64, sampler=train_sampler)
    valid_loader = DataLoader(train_val_dataset, batch_size=64, sampler=valid_sampler)

    return train_loader, valid_loader

def get_test_loader(dataset_path):
    mat = loadmat(dataset_path)

    x_mat = mat["Xtest"]
    y_mat = mat["Ytest"]

    x = torch.tensor(x_mat).permute(0, 3, 2, 1).double()
    y = torch.tensor(y_mat).double()

    test_dataset = TensorDataset(x, y)
    data_loader = DataLoader(test_dataset, batch_size=64)

    return data_loader