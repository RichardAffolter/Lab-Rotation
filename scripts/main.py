# %%
import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import time
import torch.optim as optim
from tqdm import tqdm
#from sklearn.preprocessing import LabelEncoder
#from torchinfo import summary

from gln_model import *
# %% set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
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







#%%
#path_name = "/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/Xy_toy_train.hdf5"
path_name = "/home/richard/labrotation/data/Xy_toy_train.hdf5"
# %%
train_dataset = SNPDataset(data_path=path_name)
train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True, drop_last=True)

# %%
# Hyperparameters
learning_rate = 1e-3
batch_size = 32
load_model = True

# %%
model = GLN(in_features=100, num_classes=1,num_residual_blocks=2).to(device=device)

#model.load_state_dict(torch.load('my_model3'))
# %%
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# %%
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))
# %%
for epoch in range(1):
    losses = []
    t0 = time.time()

    for batch_idx, (data,targets) in enumerate(tqdm(train_loader)):
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
    elapsed_time = (time.time() - t0)/60
    if epoch % 1 ==0:
        print(f'loss at epoch {epoch} was {mean_loss:.5f}')
        print(f'epoch took {elapsed_time:.1f} minutes')
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint)



