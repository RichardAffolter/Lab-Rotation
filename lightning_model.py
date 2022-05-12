#%%
import os
import wandb
import pytorch_lightning as pl

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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from gln_model import *

from torchmetrics.functional import r2_score

# %%
class SNPDataset(Dataset):
    def __init__(self, data_path):
        self.hf = h5py.File(data_path, 'r')
        self.X = self.hf['X/SNP']
        self.Environment = self.hf['X/Environment']
        self.y = self.hf['y/Height']
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        Environment = self.Environment[index]
        
        return X, y, Environment
    
    def dims(self):
        return (self.X.shape[0], self.X.shape[1], self.Environment.shape[1])



# %%
class lightning_gln(pl.LightningModule):
    def __init__(self, in_features, in_envs, num_classes, num_residual_blocks=2, m1=2, m2=2, C=4, num_predictor_blocks=4, lr=1e-1):
        super().__init__()
        self.gln = GLN(in_features, in_envs, num_classes, num_residual_blocks, m1, m2, C, num_predictor_blocks)
        self.lr = lr
        self.predictions = []
        self.truth = []

    def forward(self, x, y):
        return self.gln(x.float(),y.float())

    def training_step(self, batch, batch_idx):
        data, targets, Environment = batch
        targets = targets.flatten().float()
        scores = self.forward(data, Environment).flatten()
        loss = F.mse_loss(scores, targets)
        self.log("train_step_MSE", loss, on_step=False, on_epoch=True)
        return loss
        
   
    def validation_step(self, batch, batch_idx):
        data, targets, Environment = batch
        targets = targets.flatten().float()
        scores = self.forward(data, Environment).flatten()
        MSE = F.mse_loss(scores, targets)
        MAE = F.l1_loss(scores, targets)
        self.log('val_MSE', MSE, on_epoch=True)
        self.log('val_MAE', MAE, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        data, targets, Environment = batch
        targets = targets.flatten().float()
        scores = self.forward(data, Environment).flatten()
        MSE = F.mse_loss(scores, targets)
        MAE = F.l1_loss(scores, targets)
        r2 =  r2_score(scores, targets)
        self.log('test_MSE', MSE, on_step=False, on_epoch=True)
        self.log('test_MAE', MAE, on_step=False, on_epoch=True)
        self.log('test_r2',   r2, on_step=False, on_epoch=True)

        self.predictions.extend(scores.tolist())
        self.truth.extend(targets.tolist())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma = 0.5, verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5, verbose=True)
        return [optimizer], [lr_scheduler]

#%%
batch_size = 64
train_path = "/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/age_sex_env/Xy_train_10k_simple_env.hdf5"
test_path  = "/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/age_sex_env/Xy_val_10k_simple_env.hdf5"
train_dataset = SNPDataset(data_path=train_path)
test_dataset  = SNPDataset(data_path=test_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=6)
val_loader   = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=6)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=6)
# %%
if __name__ == '__main__':
    #pl.seed_everything(42, workers=True)
    wandb_logger = WandbLogger(project="height prediction")
    #early_stopping_callback = EarlyStopping(monitor='val_MAE',mode='min',min_delta=0.1,patience=2)
    trainer = pl.Trainer(fast_dev_run=False,
            accelerator="gpu",
            max_epochs=20,
            enable_progress_bar=True,
            logger=wandb_logger,
            check_val_every_n_epoch=1)
    model = lightning_gln(in_features=train_dataset.dims()[1], in_envs= train_dataset.dims()[2], num_classes=1,num_residual_blocks=2)
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)
    #print(checkpoint_callback.best_model_score.cpu())
    #model = model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    results = trainer.test(model, dataloaders=test_loader) #results dict
    np.savetxt("simple_env_predictions.csv",
           model.predictions,
           delimiter =", ",
           fmt ='% s')
     
    np.savetxt("simple_env_ground_truth.csv",
           model.truth,
           delimiter =", ",
           fmt ='% s')
#%%
