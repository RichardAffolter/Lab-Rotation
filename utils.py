# -------------------------------------------------------------------
# UKBiobank - height prediction project
#
# Code for generating the dataframes for prediction of adult height based on SNPs selected from preprocessing and weight at birth
#
# L. Bourguignon
# 05.01.2021
# -------------------------------------------------------------------

# ---- Load packages ---- #
import numpy as np
import pandas as pd
import h5py as h5
import torch
import csv
from sklearn.model_selection import train_test_split

# ---- Declare paths --- #
home_data_path = '/home/madamer/height-prediction/data/'
data_path = '/local0/scratch/madamer/height-prediction/'
hdf5_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/'
hdf5_path_50k = '/links/groups/borgwardt/Data/UKBiobank/genotype_500k/EGAD00010001497/hdf5/'
hdf5_path_big = '/links/groups/borgwardt/Data/UKBiobank/genotype_500k/EGAD00010001497/hdf5/'
csv_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/'

# ---- Helper classes ---- #

class Height_Statistics():
    def __init__(self):
        self.means = {'male':175.67252960367327,'female':162.49780160116387}
        self.stds = {'male':6.843439632761728,'female':6.303442466347971}
        self.age_reg_coeffs = {'b_1':0.021869934349245713, 'b_0':-42.680572362398614}

        # Old (calculated on train AND val set)
        # self.means = {'male':175.61737227110672,'female':162.4423303666864}
        # self.stds = {'male':6.847034853772383,'female':6.312158412031903}
        # self.age_reg_coeffs = {'b_1':0.02185025362375086, 'b_0':-42.64027880830227}

    def calculate_height(self,height,sex,age):
        correction_height = age*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0']
        height = height + correction_height
        female_idx = torch.where(sex==0)
        male_idx = torch.where(sex==1)
        height[female_idx] = height[female_idx]*self.stds['female'] + self.means['female']
        height[male_idx] = height[male_idx]*self.stds['male'] + self.means['male']
        return height

class Height_Statistics_np():
    def __init__(self):
        self.means = {'male':175.67252960367327,'female':162.49780160116387}
        self.stds = {'male':6.843439632761728,'female':6.303442466347971}
        self.age_reg_coeffs = {'b_1':0.021869934349245713, 'b_0':-42.680572362398614}

        # Old (calculated on train AND val set)
        # self.means = {'male':175.61737227110672,'female':162.4423303666864}
        # self.stds = {'male':6.847034853772383,'female':6.312158412031903}
        # self.age_reg_coeffs = {'b_1':0.02185025362375086, 'b_0':-42.64027880830227}

    def calculate_height(self,height,sex,age):
        correction_height = age*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0']
        height = height + correction_height
        female_idx = np.where(sex==0)[0]
        male_idx = np.where(sex==1)[0]
        height[female_idx] = height[female_idx]*self.stds['female'] + self.means['female']
        height[male_idx] = height[male_idx]*self.stds['male'] + self.means['male']
        return height

class Height_Score():
    def __init__(self):
        self.means = {'male':175.67252960367327,'female':162.49780160116387}
        self.stds = {'male':6.843439632761728,'female':6.303442466347971}
        self.age_reg_coeffs = {'b_1':0.021869934349245713, 'b_0':-42.680572362398614}

    def calculate_score(self,df):
        male_df = df[df['"31-0.0"']==1].copy()
        female_df = df[df['"31-0.0"']==0].copy()
        male_df.loc[:,'score'] = (male_df['"50-0.0"']-self.means['male'])/self.stds['male']
        female_df.loc[:,'score'] = (female_df['"50-0.0"']-self.means['female'])/self.stds['female']
        df = pd.concat([male_df,female_df],axis=0).sort_values('id')
        df.loc[:,'score'] = df['score']-(df['"34-0.0"']*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0'])
        return df.loc[:,['id','score']]

class Height_Statistics_big():
    def __init__(self):
        self.means = {'male':175.6724565726849,'female':162.4978568575439}
        self.stds = {'male':6.843617921273466,'female':6.303270989924073}
        self.age_reg_coeffs = {'b_1':0.021870142331929222, 'b_0':-42.68098588591243}

    def calculate_height(self,height,sex,age):
        correction_height = age*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0']
        height = height + correction_height
        female_idx = torch.where(sex==0)
        male_idx = torch.where(sex==1)
        height[female_idx] = height[female_idx]*self.stds['female'] + self.means['female']
        height[male_idx] = height[male_idx]*self.stds['male'] + self.means['male']
        return height

class Height_Score_big():
    def __init__(self):
        self.means = {'male':175.6724565726849,'female':162.4978568575439}
        self.stds = {'male':6.843617921273466,'female':6.303270989924073}
        self.age_reg_coeffs = {'b_1':0.021870142331929222, 'b_0':-42.68098588591243}

    def calculate_score(self,df):
        male_df = df[df['"31-0.0"']==1].copy()
        female_df = df[df['"31-0.0"']==0].copy()
        male_df.loc[:,'score'] = (male_df['"50-0.0"']-self.means['male'])/self.stds['male']
        female_df.loc[:,'score'] = (female_df['"50-0.0"']-self.means['female'])/self.stds['female']
        df = pd.concat([male_df,female_df],axis=0).sort_values('id')
        df.loc[:,'score'] = df['score']-(df['"34-0.0"']*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0'])
        return df.loc[:,['id','score']]
# ---- Functions ---- #

def convert_to_csv(file):
    data = file['genotype/matrix']
    df = pd.DataFrame(data[:, :])
    id_list = get_id(file)
    print(id_list.head())
    df['id'] = id_list[0]
    print(f'SNP dataframe created for {file}')
    return df, id_list

def get_id(file):
    id_sample_hf = file['sample_info/iid']
    id_sample = pd.DataFrame(np.array(id_sample_hf))
    # print(id_sample)
    # for i in range(0, id_sample.shape[0]):
    #    id_sample.iloc[i, 0] = id_sample.iloc[i, 0].decode("utf-8")
    id_sample = id_sample.astype(int)
    # id_train, id_test = train_test_split(id_sample[0].tolist(), test_size=0.33, random_state=42)
    print(f'IDs created for {file}')
    return id_sample

def load_datafield(number):
    raw_df = pd.read_csv(data_path + f'{number}.csv', engine='python', quoting=csv.QUOTE_NONE)
    data = raw_df.replace('"', '', regex=True)
    data['"eid"'] = data['"eid"'].astype(int)
    data.set_index('"eid"')
    data = data.rename(columns={'"eid"': 'id'})
    subset = data.iloc[:, 0:2]
    print(f'Datafield {number} loaded')
    return subset

def generate_df_height_prediction(df_snp, id_df, scenario):

    df_height_pred = pd.DataFrame()
    df_height_pred['id'] = id_df[0]
    df_height_pred = pd.merge(df_height_pred, df_height)

    if scenario == 'weight':
        df_height_pred = pd.merge(df_height_pred, df_weight_birth)
    elif scenario == 'snp' :
    	df_height_pred = pd.merge(df_height_pred, df_snp)
    elif scenario == 'weight_snp' :
    	df_height_pred = pd.merge(df_height_pred, df_weight_birth)
    	df_height_pred = pd.merge(df_height_pred, df_snp)


    return df_height_pred
