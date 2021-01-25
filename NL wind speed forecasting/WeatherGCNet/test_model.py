import torch
from model_nl import Model
from data_loader_wind_nl import *
import torch.nn.functional as F
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dataset_path", required=True, help="path to the .pkl dataset, e.g. 'data/dataset.pkl'")
ap.add_argument("-it", "--input_timesteps", type=int, required=True, help="number of timesteps to be included in the input")
ap.add_argument("-pt", "--predict_timestep", type=int, required=True, help="number of timesteps (hours) ahead for the prediction")
ap.add_argument("-mp", "--model_path", required=True, help="path to the .pt model, e.g. 'trained_models/best_model_20201008_112302.pt'")
args = vars(ap.parse_args())

dataset_path = args["dataset_path"]
input_timesteps = args["input_timesteps"]
predict_timestep = args["predict_timestep"]
model_path = args["model_path"]

print("it", input_timesteps, "pt", predict_timestep, "\nmp", model_path, "\n")

torch.manual_seed(123)
torch.set_printoptions(precision=20)

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(dev)

# from scaler.pkl (ralted to dataset.pkl) - WIND SPEED
scaler_min = torch.tensor([0.000e+00]).to(dev)
scaler_max = torch.tensor([240.]).to(dev)

loss_func = F.l1_loss
loss_func_2 = F.mse_loss

best_model = Model()
best_model = best_model.double().to(dev) 
best_model.load_state_dict(torch.load(model_path))

test_dl = get_test_loader(dataset_path, input_timesteps, predict_timestep)

best_model.eval()
with torch.no_grad():
    test_loss = 0.0
    test_loss_2 = 0.0
    test_loss_2_scaled = 0.0
    test_num = 0
    
    for xb, yb in test_dl:
        pred = best_model(xb.to(dev))
        pred_unscaled = pred * (scaler_max - scaler_min) + scaler_min
        yb_unscaled = yb.to(dev) * (scaler_max - scaler_min) + scaler_min

        batch_test_loss = loss_func(pred_unscaled, yb_unscaled, reduction='none')
        batch_test_loss_2 = loss_func_2(pred_unscaled, yb_unscaled, reduction='none')
        batch_test_loss_2_scaled = loss_func_2(pred, yb.to(dev), reduction='none')

        test_loss += torch.sum(batch_test_loss, dim=0)
        test_loss_2 += torch.sum(batch_test_loss_2, dim=0)
        test_loss_2_scaled += torch.sum(batch_test_loss_2_scaled, dim=0)
        test_num += len(xb)

test_loss /= test_num # average of all samples
print("\nTest MAE loss of the model:", test_loss)
print("Average:", torch.mean(test_loss), "\n")

test_loss_2 /= test_num # average of all samples
print("Test MSE loss of the model:", test_loss_2)
print("Average:", torch.mean(test_loss_2), "\n")

test_loss_2_scaled /= test_num # average of all samples
print("Test MSE loss (scaled data) of the model:", test_loss_2_scaled)
print("Average:", torch.mean(test_loss_2_scaled), "\n")