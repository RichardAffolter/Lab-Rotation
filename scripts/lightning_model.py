#%%
import os
import wandb
import pytorch_lightning as pl
from utils import Height_Statistics,Height_Statistics_big

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

from pytorch_lightning.loggers import WandbLogger

#from sklearn.preprocessing import LabelEncoder
#from torchinfo import summary

from gln_model import *

# %%
class SNPDataset(Dataset):
    def __init__(self, data_path):
        self.hf = h5py.File(data_path, 'r')
        self.X = self.hf['X/SNP']
        self.Age = self.hf['X/Age']
        self.Sex = self.hf['X/Sex']
        self.y = self.hf['y/Height']
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        Sex = self.Sex[index]
        Age = self.Age[index]
        
        return X, y, Sex, Age
    
    def dims(self):
        return self.X.shape



# %%
class lightning_gln(pl.LightningModule):
    def __init__(self, in_features, num_classes, num_residual_blocks=2, m1=2, m2=2, C=4, num_predictor_blocks=4, lr=1e-3):
        super().__init__()
        self.gln = GLN(in_features, num_classes, num_residual_blocks, m1, m2, C, num_predictor_blocks)
        self.stats = Height_Statistics_big()
        self.lr = lr

    def forward(self, x):
        return self.gln(x)

    def training_step(self, batch, batch_idx):
        data, targets, Sex, Age = batch
        targets = targets.flatten()
        scores = self.forward(data).flatten()
        loss = F.mse_loss(scores, targets)
        self.log("val_step_MSE", loss, on_step=False, on_epoch=True)
        return loss
        
   
    def validation_step(self, batch, batch_idx):
        data, targets, Sex, Age = batch
        targets = targets.flatten()
        scores = self.forward(data).flatten()
        pred_height = self.stats.calculate_height(height=scores, sex=Sex, age=Age)
        true_height = self.stats.calculate_height(height=targets, sex=Sex, age=Age)
        MSE = F.mse_loss(true_height, pred_height)
        MAE = F.l1_loss(true_height, pred_height)
        self.log('val_MSE', MSE, on_epoch=True)
        self.log('val_MAE', MAE, on_epoch=True)


    def test_step(self, batch, batch_idx):
        data, targets, Sex, Age = batch
        targets = targets.flatten()
        scores = self.forward(data).flatten()
        pred_height = self.stats.calculate_height(height=scores, sex=Sex, age=Age)
        true_height = self.stats.calculate_height(height=targets, sex=Sex, age=Age)
        MSE = F.mse_loss(true_height, pred_height)
        MAE = F.l1_loss(true_height, pred_height)
        self.log('val_MSE', MSE, on_step=False, on_epoch=True)
        self.log('val_MAE', MAE, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    

#%%
batch_size = 64
#path_name = "/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/Xy_toy_train.hdf5"
train_path = "/home/richard/labrotation/data/Xy_toy_train.hdf5"
test_path =  "/home/richard/labrotation/data/Xy_toy_val.hdf5"
train_dataset = SNPDataset(data_path=train_path)
test_dataset  = SNPDataset(data_path=test_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=4)
val_loader   = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
# %%
#if __name__ == '__main__':
pl.seed_everything(42, workers=True)
trainer = pl.Trainer(fast_dev_run=False, 
                    max_epochs=1, 
                    check_val_every_n_epoch=1)
model = lightning_gln(in_features=train_dataset.dims()[1], num_classes=1,num_residual_blocks=2)

trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)
results = trainer.test(dataloaders=test_loader) #results dict

#%%
