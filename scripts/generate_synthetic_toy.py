import argparse
import os

import h5py as h5
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed
import fire

# from src.utils import *
DATA_PATH = '/home/michael/ETH/data/HeightPrediction/'

class model:
    def __init__(self,dim,alpha=1.,beta=1.,gamma=0.1,epi_fraction=1.0):
        self.dim = dim
        self.first_order_weights = np.random.randn(self.dim)

        # Second order weights with epi_fraction
        n_epi = int(self.dim*(self.dim-1)/2)
        tmp_weights = np.random.randn(n_epi)
        tmp_mask = np.arange(n_epi)
        np.random.shuffle(tmp_mask)
        tmp_mask = tmp_mask[:int((1-epi_fraction)*n_epi)]
        tmp_weights[tmp_mask] = 0
        self.second_oder_weights = tmp_weights

        self.bias = 0.
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def forward(self,x):
        x_squared = np.outer(x,x)
        x_squared_idx = np.triu_indices(x_squared.shape[0],k=1)#(x_squared - np.diag(np.diag(x_squared))).flatten()
        x_squared = x_squared[x_squared_idx].flatten()
        lin = np.multiply(self.first_order_weights,x).sum()
        sq = np.multiply(self.second_oder_weights,x_squared).sum()
        if self.gamma != 0:
            ratio = (self.beta*lin)/(self.gamma*sq)
        else:
            ratio = np.inf
        return self.alpha*(self.beta*lin + self.gamma*sq + self.bias),ratio

class marchini_model_1:
    def __init__(self,dim,alpha=1.,beta=1.,gamma=0.1,epi_fraction=1.0):
        self.dim = dim
        self.first_order_weights = np.random.randn(self.dim)

        # Second order weights with epi_fraction
        n_epi = int(self.dim*(self.dim-1)/2)
        tmp_weights = np.random.randn(n_epi)
        tmp_mask = np.arange(n_epi)
        np.random.shuffle(tmp_mask)
        tmp_mask = tmp_mask[:int((1-epi_fraction)*n_epi)]
        tmp_weights[tmp_mask] = 0
        self.second_oder_weights = tmp_weights

        self.parameters = 1.+np.random.uniform(size=int(self.dim*(self.dim-1)/2))
        self.bias = 0.
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def forward(self,x):
        # More elegant way?
        f = np.zeros(int(x.shape[0]*(x.shape[0]-1)/2))
        count = 0
        for x1 in np.arange(x.shape[0]-1):
            for x2 in np.arange(x.shape[0])[x1+1:]:
                f[count] = x[x1]+x[x2]
                count = count+1
        x_squared = np.power(self.parameters,f)
        lin = np.multiply(self.first_order_weights,x).sum()
        sq = np.multiply(self.second_oder_weights,x_squared).sum()
        if self.gamma != 0:
            ratio = (self.beta*lin)/(self.gamma*sq)
        else:
            ratio = np.inf
        return self.alpha*(self.beta*lin + self.gamma*sq + self.bias), ratio

class marchini_model_2:
    def __init__(self,dim,alpha=1.,beta=1.,gamma=0.1,epi_fraction=1.0):
        self.dim = dim
        self.first_order_weights = np.random.randn(self.dim)

        # Second order weights with epi_fraction
        n_epi = int(self.dim*(self.dim-1)/2)
        tmp_weights = np.random.randn(n_epi)
        tmp_mask = np.arange(n_epi)
        np.random.shuffle(tmp_mask)
        tmp_mask = tmp_mask[:int((1-epi_fraction)*n_epi)]
        tmp_weights[tmp_mask] = 0
        self.second_oder_weights = tmp_weights

        self.parameters = 1.+np.random.uniform(size=int(self.dim*(self.dim-1)/2))
        self.bias = 0.
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def forward(self,x):
        # More elegant way?
        f = np.zeros(int(x.shape[0]*(x.shape[0]-1)/2))
        count = 0
        for x1 in np.arange(x.shape[0]-1):
            for x2 in np.arange(x.shape[0])[x1+1:]:
                f[count] = x[x1]*x[x2]
                count = count+1
        x_squared = np.power(self.parameters,f)
        lin = np.multiply(self.first_order_weights,x).sum()
        sq = np.multiply(self.second_oder_weights,x_squared).sum()
        if self.gamma != 0:
            ratio = (self.beta*lin)/(self.gamma*sq)
        else:
            ratio = np.inf
        return self.alpha*(self.beta*lin + self.gamma*sq + self.bias), ratio


class marchini_model_3:
    def __init__(self,dim,alpha=1.,beta=1.,gamma=0.1,epi_fraction=1.0):
        self.dim = dim
        self.first_order_weights = np.random.randn(self.dim)

        # Second order weights with epi_fraction
        n_epi = int(self.dim*(self.dim-1)/2)
        tmp_weights = np.random.randn(n_epi)
        tmp_mask = np.arange(n_epi)
        np.random.shuffle(tmp_mask)
        tmp_mask = tmp_mask[:int((1-epi_fraction)*n_epi)]
        tmp_weights[tmp_mask] = 0
        self.second_oder_weights = tmp_weights

        self.parameters = 1.+np.random.uniform(size=int(self.dim*(self.dim-1)/2))
        self.bias = 0.
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def forward(self,x):
        # More elegant way?
        f = np.zeros(int(x.shape[0]*(x.shape[0]-1)/2))
        count = 0
        for x1 in np.arange(x.shape[0]-1):
            for x2 in np.arange(x.shape[0])[x1+1:]:
                f[count] = np.sign(x[x1]*x[x2])
                count = count+1
        x_squared = np.power(self.parameters,f)
        lin = np.multiply(self.first_order_weights,x).sum()
        sq = np.multiply(self.second_oder_weights,x_squared).sum()
        if self.gamma != 0:
            ratio = (self.beta*lin)/(self.gamma*sq)
        else:
            ratio = np.inf
        return self.alpha*(self.beta*lin + self.gamma*sq + self.bias), ratio

def embed_dominances(data,dominances):
    snps = np.array(data)
    heterozygous = []
    for s in tqdm(snps):
        tmp = np.zeros_like(s)
        tmp[s==1] = 1
        heterozygous.append(tmp)
    heterozygous = np.array(heterozygous)
    return snps + np.multiply(heterozygous,dominances)


def from_hdf5(beta):
    train_file = h5.File(DATA_PATH+'Xy_toy_train.hdf5','r+')
    val_file = h5.File(DATA_PATH+'Xy_toy_val.hdf5','r+')

    train_data = train_file['X/SNP']
    val_data = val_file['X/SNP']

    syn_model = model(int(1e4),beta=beta)

    for data in [train_data,val_data]:
        syn = []
        out = Parallel(n_jobs=128,verbose=1)(delayed(syn_model.squared)(d) for d in data)
        syn.append(d)
        syn = np.array(syn).flatten()
        print(np.mean(syn),np.std(syn))
        data['y'].create_dataset("y_syn",data=syn)

def squared(beta,gamma,epi_fraction):
    np.random.seed(47)

    train_file = h5.File(DATA_PATH+'Xy_toy_train.hdf5','r+')
    val_file = h5.File(DATA_PATH+'Xy_toy_val.hdf5','r+')

    train_data = train_file['X/SNP']
    val_data = val_file['X/SNP']

    num_snps = train_data.shape[1]
    model_cl = model(num_snps,beta=beta,gamma=gamma)
    snp_probs = np.mean(train_data,axis=0)/2

    # print(f'Minor allele frequencies: {snp_probs}')


    ##### TRAIN ########
    y_syn = []
    eff_sq = []

    for d in tqdm(train_data):
        syn,sq_effect = model_cl.forward(d)
        y_syn.append(syn)
        eff_sq.append(sq_effect)

    y_syn = np.array(y_syn).flatten()
    print(np.mean(y_syn),np.std(y_syn),np.mean(sq_effect))

    if 'y_squared' in train_file['y'].keys():
        del train_file['y/y_squared']
    train_file['y'].create_dataset('y_squared',data=y_syn)
    train_file.close()

    ##### TEST ########
    y_syn = []
    eff_sq = []

    for d in tqdm(val_data):
        syn,sq_effect = model_cl.forward(d)
        y_syn.append(syn)
        eff_sq.append(sq_effect)

    y_syn = np.array(y_syn).flatten()
    print(np.mean(y_syn),np.std(y_syn),np.mean(sq_effect))

    if 'y_squared' in val_file['y'].keys():
        del val_file['y/y_squared']
    val_file['y'].create_dataset('y_squared',data=y_syn)
    val_file.close()

def Cordell(beta,gamma,epi_fraction):
    np.random.seed(47)

    train_file = h5.File(DATA_PATH+'Xy_toy_train.hdf5','r+')
    val_file = h5.File(DATA_PATH+'Xy_toy_val.hdf5','r+')

    train_data = train_file['X/SNP']
    val_data = val_file['X/SNP']

    num_snps = train_data.shape[1]


    # Second order weights with epi_fraction
    tmp_dominances = np.random.uniform(low=-1.,high=1.,size=(1,num_snps))
    tmp_mask = np.arange(num_snps)
    np.random.shuffle(tmp_mask)
    tmp_mask = tmp_mask[:int((1-epi_fraction)*num_snps)]
    tmp_dominances[0][tmp_mask] = 0
    dominances = tmp_dominances

    model_cl = model(num_snps,beta=beta,gamma=gamma)
    snp_probs = np.mean(train_data,axis=0)/2

    # print(f'Minor allele frequencies: {snp_probs}')

    ##### TRAIN ########
    snps = embed_dominances(train_data,dominances)
    snp_probs_d = np.mean(snps,axis=0)/2
    # print(f'New minor allele frequencies: {snp_probs_d}')

    y_syn = []
    eff_sq = []

    for d in tqdm(snps):
        syn,sq_effect = model_cl.forward(d)
        y_syn.append(syn)
        eff_sq.append(sq_effect)

    y_syn = np.array(y_syn).flatten()
    print(np.mean(y_syn),np.std(y_syn),np.mean(sq_effect))

    if 'y_Cordell' in train_file['y'].keys():
        del train_file['y/y_Cordell']
    train_file['y'].create_dataset('y_Cordell',data=y_syn)
    train_file.close()

    ##### TEST ########
    snps = embed_dominances(val_data,dominances)
    snp_probs_d = np.mean(snps,axis=0)/2
    # print(f'New minor allele frequencies: {snp_probs_d}')

    y_syn = []
    eff_sq = []

    for d in tqdm(snps):
        syn,sq_effect = model_cl.forward(d)
        y_syn.append(syn)
        eff_sq.append(sq_effect)

    y_syn = np.array(y_syn).flatten()
    print(np.mean(y_syn),np.std(y_syn),np.mean(sq_effect))

    if 'y_Cordell' in val_file['y'].keys():
        del val_file['y/y_Cordell']
    val_file['y'].create_dataset('y_Cordell',data=y_syn)
    val_file.close()

def Marchini_1(beta,gamma,epi_fraction):
    np.random.seed(47)

    train_file = h5.File(DATA_PATH+'Xy_toy_train.hdf5','r+')
    val_file = h5.File(DATA_PATH+'Xy_toy_val.hdf5','r+')

    train_data = train_file['X/SNP']
    val_data = val_file['X/SNP']

    num_snps = train_data.shape[1]
    model_cl = marchini_model_1(num_snps,beta=beta,gamma=gamma,epi_fraction=epi_fraction)

    snp_probs = np.mean(train_data,axis=0)/2
    # print(f'Minor allele frequencies: {snp_probs}')

    ##### TRAIN ########
    y_syn = []
    eff_sq = []

    for d in tqdm(train_data):
        syn,sq_effect = model_cl.forward(d)
        y_syn.append(syn)
        eff_sq.append(sq_effect)

    y_syn = np.array(y_syn).flatten()
    print(np.mean(y_syn),np.std(y_syn),np.mean(sq_effect))

    if 'y_Marchini_1' in train_file['y'].keys():
        del train_file['y/y_Marchini_1']
    train_file['y'].create_dataset('y_Marchini_1',data=y_syn)
    train_file.close()

    ##### TEST ########
    y_syn = []
    eff_sq = []

    for d in tqdm(val_data):
        syn,sq_effect = model_cl.forward(d)
        y_syn.append(syn)
        eff_sq.append(sq_effect)

    y_syn = np.array(y_syn).flatten()
    print(np.mean(y_syn),np.std(y_syn),np.mean(sq_effect))

    if 'y_Marchini_1' in val_file['y'].keys():
        del val_file['y/y_Marchini_1']
    val_file['y'].create_dataset('y_Marchini_1',data=y_syn)
    val_file.close()

def Marchini_2(beta,gamma,epi_fraction):
    np.random.seed(47)

    train_file = h5.File(DATA_PATH+'Xy_toy_train.hdf5','r+')
    val_file = h5.File(DATA_PATH+'Xy_toy_val.hdf5','r+')

    train_data = train_file['X/SNP']
    val_data = val_file['X/SNP']

    num_snps = train_data.shape[1]
    model_cl = marchini_model_2(num_snps,beta=beta,gamma=gamma,epi_fraction=epi_fraction)

    snp_probs = np.mean(train_data,axis=0)/2
    # print(f'Minor allele frequencies: {snp_probs}')

    ##### TRAIN ########
    y_syn = []
    eff_sq = []

    for d in tqdm(train_data):
        syn,sq_effect = model_cl.forward(d)
        y_syn.append(syn)
        eff_sq.append(sq_effect)

    y_syn = np.array(y_syn).flatten()
    print(np.mean(y_syn),np.std(y_syn),np.mean(sq_effect))

    if 'y_Marchini_2' in train_file['y'].keys():
        del train_file['y/y_Marchini_2']
    train_file['y'].create_dataset('y_Marchini_2',data=y_syn)
    train_file.close()

    ##### TEST ########
    y_syn = []
    eff_sq = []

    for d in tqdm(val_data):
        syn,sq_effect = model_cl.forward(d)
        y_syn.append(syn)
        eff_sq.append(sq_effect)

    y_syn = np.array(y_syn).flatten()
    print(np.mean(y_syn),np.std(y_syn),np.mean(sq_effect))

    if 'y_Marchini_2' in val_file['y'].keys():
        del val_file['y/y_Marchini_2']
    val_file['y'].create_dataset('y_Marchini_2',data=y_syn)
    val_file.close()

def Marchini_3(beta,gamma,epi_fraction):
    np.random.seed(47)

    train_file = h5.File(DATA_PATH+'Xy_toy_train.hdf5','r+')
    val_file = h5.File(DATA_PATH+'Xy_toy_val.hdf5','r+')

    train_data = train_file['X/SNP']
    val_data = val_file['X/SNP']

    num_snps = train_data.shape[1]
    model_cl = marchini_model_3(num_snps,beta=beta,gamma=gamma,epi_fraction=epi_fraction)

    snp_probs = np.mean(train_data,axis=0)/2
    # print(f'Minor allele frequencies: {snp_probs}')

    ##### TRAIN ########
    y_syn = []
    eff_sq = []

    for d in tqdm(train_data):
        syn,sq_effect = model_cl.forward(d)
        y_syn.append(syn)
        eff_sq.append(sq_effect)

    y_syn = np.array(y_syn).flatten()
    print(np.mean(y_syn),np.std(y_syn),np.mean(sq_effect))

    if 'y_Marchini_3' in train_file['y'].keys():
        del train_file['y/y_Marchini_3']
    train_file['y'].create_dataset('y_Marchini_3',data=y_syn)
    train_file.close()

    ##### TEST ########
    y_syn = []
    eff_sq = []

    for d in tqdm(val_data):
        syn,sq_effect = model_cl.forward(d)
        y_syn.append(syn)
        eff_sq.append(sq_effect)

    y_syn = np.array(y_syn).flatten()
    print(np.mean(y_syn),np.std(y_syn),np.mean(sq_effect))

    if 'y_Marchini_3' in val_file['y'].keys():
        del val_file['y/y_Marchini_3']
    val_file['y'].create_dataset('y_Marchini_3',data=y_syn)
    val_file.close()

def generate(model=None,beta=1.,gamma=0.1,epi_fraction=0.5):
    if model is not None:
        eval(f'{model}(beta={beta},gamma={gamma},epi_fraction={epi_fraction})')
    else:
        print('Name a model (squared, Cordell, Marchini_1-3)')

if __name__ == "__main__":
    fire.Fire(generate)
