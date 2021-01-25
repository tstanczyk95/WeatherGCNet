from scipy.io import loadmat
from model_nl import Model
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import time
from datetime import datetime
from data_loader_wind_nl import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dataset_path", required=True, help="path to the .pkl dataset, e.g. 'data/dataset.pkl'")
ap.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
ap.add_argument("-it", "--input_timesteps", type=int, required=True, help="number of timesteps to be included in the input")
ap.add_argument("-pt", "--predict_timestep", type=int, required=True, help="number of timesteps (hours) ahead for the prediction")
args = vars(ap.parse_args())

dataset_path = args["dataset_path"]
epochs = args["epochs"]
input_timesteps = args["input_timesteps"]
predict_timestep = args["predict_timestep"]

torch.manual_seed(123)
torch.set_printoptions(precision=20)

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(dev)

#from scaler.pkl (ralted to dataset.pkl) - WIND SPEED
scaler_min = torch.tensor([0.000e+00]).to(dev)
scaler_max = torch.tensor([240.]).to(dev)

model = Model()
model = model.double().to(dev)

train_dl, valid_dl = get_train_valid_loader(dataset_path, input_timesteps, predict_timestep)

loss_func = F.l1_loss
loss_func_2 = F.mse_loss

opt = optim.Adam(model.parameters())

best_mean_valid_loss = 1e4

now = datetime.now()
date_time_string = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + "_" + str(now.hour + 2).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

print("Date time string: ", date_time_string)

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb.to(dev))
        loss = loss_func(pred, yb.to(dev))

        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        valid_num = 0
        
        for xb, yb in valid_dl:
            batch_valid_loss = loss_func(model(xb.to(dev)), yb.to(dev), reduction='none')
            valid_loss += torch.sum(batch_valid_loss, dim=0)
            valid_num += len(xb)

        valid_loss /= valid_num # average of all samples
        mean_valid_loss = torch.mean(valid_loss) # average of all outputs

        if mean_valid_loss < best_mean_valid_loss:
            torch.save(model.state_dict(), "trained_models/best_model_nl_" + date_time_string + ".pt")
            print("---Model with the following mean valid loss saved: ", mean_valid_loss.item())
            best_mean_valid_loss = mean_valid_loss


        print(epoch, mean_valid_loss)

# # # # # # # # # #

best_model = Model()
best_model = best_model.double().to(dev)
best_model.load_state_dict(torch.load("trained_models/best_model_nl_" + date_time_string + ".pt"))

test_dl = get_test_loader(dataset_path, input_timesteps, predict_timestep)

best_model.eval()
with torch.no_grad():
    test_loss = 0.0
    test_loss_2 = 0.0
    test_num = 0
    
    for xb, yb in test_dl:
        pred = best_model(xb.to(dev))
        pred_unscaled = pred * (scaler_max - scaler_min) + scaler_min
        yb_unscaled = yb.to(dev) * (scaler_max - scaler_min) + scaler_min

        batch_test_loss = loss_func(pred_unscaled, yb_unscaled, reduction='none')
        batch_test_loss_2 = loss_func_2(pred_unscaled, yb_unscaled, reduction='none')

        test_loss += torch.sum(batch_test_loss, dim=0)
        test_loss_2 += torch.sum(batch_test_loss_2, dim=0)
        test_num += len(xb)

test_loss /= test_num # average of each samples
print("\nTest MAE loss of the model:", test_loss)
print("Average:", torch.mean(test_loss), "\n")

test_loss_2 /= test_num # average of each samples
print("Test MSE loss of the model:", test_loss_2)
print("Average:", torch.mean(test_loss_2), "\n")
