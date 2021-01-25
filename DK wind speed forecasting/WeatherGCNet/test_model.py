import torch
from model import Model
from data_loader_wind_dk import *
import torch.nn.functional as F
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dataset_path", required=True, help="path to the .mat dataset, e.g. 'data/step1.mat'")
ap.add_argument("-mp", "--model_path", required=True, help="path to the .pt model, e.g. 'trained_models/best_model_20201008_112302.pt'")
args = vars(ap.parse_args())

dataset_path = args["dataset_path"]
model_path = args["model_path"]

torch.manual_seed(123)
torch.set_printoptions(precision=20)

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(dev)

# from scale1.mat (and from scale2.mat, scale3.mat, scale4.mat) - WIND SPEED
scaler = torch.tensor([28.3, 62.52179487, 25.]).to(dev)

best_model = Model()
best_model = best_model.double().to(dev) 
best_model.load_state_dict(torch.load(model_path))

loss_func = F.l1_loss
loss_func_2 = F.mse_loss

test_dl = get_test_loader(dataset_path)

best_model.eval()
with torch.no_grad():
    test_loss = 0.0
    test_loss_2 = 0.0
    test_num = 0
    
    for xb, yb in test_dl:
        batch_test_loss = loss_func(best_model(xb.to(dev)) * scaler, yb.to(dev) * scaler, reduction='none')
        batch_test_loss_2 = loss_func_2(best_model(xb.to(dev)) * scaler, yb.to(dev) * scaler, reduction='none')

        test_loss += torch.sum(batch_test_loss, dim=0)
        test_loss_2 += torch.sum(batch_test_loss_2, dim=0)
        test_num += len(xb)

test_loss /= test_num # average of all samples
print("\nTest MAE loss of the model:", test_loss)
print("Average:", torch.mean(test_loss), "\n")

test_loss_2 /= test_num # average of all samples
print("Test MSE loss of the model:", test_loss_2)
print("Average:", torch.mean(test_loss_2), "\n")