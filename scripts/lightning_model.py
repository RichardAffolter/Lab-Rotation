#%%
import os
import wandb
import pytorch_lightning as pl
#from src.utils import Height_Statistics,Height_Statistics_big
#from src.models.torchLASSO import torchLASSO
#from src.models.RowdyActivation import RowdyActivation

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

#from sklearn.preprocessing import LabelEncoder
#from torchinfo import summary

from gln_model import *

# %%
class SNPDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.hf = h5py.File(data_path, 'r')
        self.X = self.hf['X/SNP']
        self.age = self.hf['X/Age']
        self.Sex = self.hf['X/Sex']
        self.y = self.hf['y/Height']
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        Sex = self.Sex[index]
        Age = self.Age [index]
        
        if self.transform:
            X = self.transform(X)
        
        return X, y, Sex, Age
    
    def dims(self):
        return self.X.shape

learning_rate = 1e-1
batch_size = 32
load_model = False
#path_name = "/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/Xy_toy_train.hdf5"
path_name = "/home/richard/labrotation/data/Xy_toy_train.hdf5"

# %%
train_dataset = SNPDataset(data_path=path_name)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
print(train_dataset.dims()[1])
# %%
class lightning_gln(pl.LightningModule):
    def __init__(self, in_features, num_classes, num_residual_blocks=2, m1=2, m2=2, C=4, num_predictor_blocks=4):
        super().__init__()
        self.gln = GLN(in_features, num_classes, num_residual_blocks, m1, m2, C, num_predictor_blocks)
    
    def forward(self, x):
        return self.gln(x)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        data = F.one_hot(data.long(), num_classes=4)
        data = data.transpose(-1,-2)
        targets = targets.flatten()

        # forwards
        scores = self.forward(data).flatten()
        print(scores.shape)
        loss = F.l1_loss(scores, targets)
        self.log("train_loss", loss)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    

# %%
if __name__ == '__main__':
    trainer = pl.Trainer(fast_dev_run=True)
    model = lightning_gln(in_features=train_dataset.dims()[1], num_classes=1,num_residual_blocks=2)
    trainer.fit(model, train_dataloaders=train_loader)
# %%
from utils import Height_Statistics,Height_Statistics_big
# %%
batch = next(iter(train_loader))
# %%
batch[0]
# %%
batch[1]
# %%
