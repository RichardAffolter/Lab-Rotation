# %%
import os
import math
import pandas as pd
import numpy as np
import h5py
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.profiler
from tqdm import tqdm

#from sklearn.preprocessing import LabelEncoder
#from torchinfo import summary

from gln_model import *
# %% set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %% Accuracy function for categorical targets
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            predictions = torch.argmax(scores, dim=1)
            

            num_correct += (predictions==y).sum()
            num_samples += predictions.shape[0]
        #acc = float(num_correct/num_samples)*100
        #print(f' Got {num_correct} / {num_samples} with accuracy {float(num_correct/num_samples)*100}')

    model.train()
    return(num_correct,num_samples)

#%% mean absolute error for continuous targets
def get_MAE(loader, model):
    running_MAE = 0.0
    running_MSE = 0.0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            error = torch.abs(scores - y)
            squared_error = torch.square(error)
            
            runnning_mae += error
            runnning_mse += squared_error
            num_samples += scores.shape[0]
        
        MAE = running_MAE/num_samples
        MSE = running_MSE(num_samples)
        print(f' Got MAE: {MAE} and MSE: {MSE}')

    model.train()
    return(MAE,MSE)
# %%
class SNPDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.hf = h5py.File(data_path, 'r')
        self.X = self.hf['X/SNP']
        self.y = self.hf['y/Height']
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        
        if self.transform:
            X = self.transform(X)
        
        return X, y
    
    def dims(self):
        return self.X.shape

# %%
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# %%
# Hyperparameters
learning_rate = 1e-1
batch_size = 32
load_model = False
#path_name = "/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/Xy_toy_train.hdf5"
path_name = "/home/richard/labrotation/data/Xy_toy_train.hdf5"
# %%
train_dataset = SNPDataset(data_path=path_name)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)



# %%
model = GLN(in_features=train_dataset.dims()[1], num_classes=1,num_residual_blocks=2).to(device=device)

#model.load_state_dict(torch.load('my_model3'))
# %%
criterion = nn.L1Loss().cuda() if torch.cuda.is_available() else nn.L1Loss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2 ,patience=2, verbose=True)

# %%
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))
# %%
for epoch in range(1):
    losses = []
    t0 = time.time()

    for batch_idx, (data,targets) in enumerate(train_loader):
        data = F.one_hot(data.long(), num_classes=4)
        data = data.transpose(-1,-2)
        data = data.to(device=device)
        targets = targets.flatten().to(device=device)

        # forwards
        scores = model(data).flatten()
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backwards
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()
    
    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)
    elapsed_time = (time.time() - t0)/60
    if epoch % 1 ==0:
        print(f'loss at epoch {epoch} was {mean_loss:.5f}')
        print(f'epoch took {elapsed_time:.1f} minutes')
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint)



#%% profiler
def train(data,targets):
    data = F.one_hot(data.long(), num_classes=4)
    data = data.transpose(-1,-2)
    data = data.to(device=device)
    targets = targets.flatten().to(device=device)

    # forwards
    scores = model(data).flatten()
    loss = criterion(scores, targets)

    # backwards
    optimizer.zero_grad()
    loss.backward()

    # gradient descent
    optimizer.step()

#%%
if False:
        
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/toy'),
            record_shapes=False,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for step, (data,targets) in enumerate(train_loader):
            if step >= (1 + 1 + 3) * 2:
                break
            train(data,targets)
            prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.
    # %%

from torchinfo import summary
# %%
summary(model)