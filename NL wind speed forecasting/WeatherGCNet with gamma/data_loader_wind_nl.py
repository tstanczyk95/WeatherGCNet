from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from dataset_wind_nl import dataset_wind_nl


def get_train_valid_loader(dataset_path, inputTimesteps, predictTimestep):
    train_val_dataset = dataset_wind_nl(dataset_path=dataset_path, inputTimesteps=inputTimesteps, predictTimestep=predictTimestep, train=True)

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


def get_test_loader(dataset_path, inputTimesteps, predictTimestep):
    test_dataset = dataset_wind_nl(
        dataset_path=dataset_path, inputTimesteps=inputTimesteps, predictTimestep=predictTimestep, train=False	
    )

    data_loader = DataLoader(test_dataset, batch_size=64)

    return data_loader