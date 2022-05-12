import h5py as h5
import pandas as pd
import numpy as np
import os
import logging
from glob import glob
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_environment_data(data_path,out_path=None):
    if out_path is None:
        out_path = data_path

    files = glob(os.path.join(data_path,'*.csv'))
    df_list = [pd.read_csv(f,index_col=0) for f in files]
    df = pd.concat(df_list,axis=1)

    # Select columns which have a small number of missing values
    # The 0.11 is a sort of optimum which I chose through trial and error. Feel free to adjust
    cols_to_select = df.isna().mean()[df.isna().mean() < 0.11].index
    X = df.loc[:,cols_to_select].dropna(axis=0)

    # Load the indices of the train/val samples. Extract those from Lucie's preprocessed files.
    # I also upload them to the server
    train_idxs = np.loadtxt(os.path.join('/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/',
        'train_samples.txt'))
    val_idxs = np.loadtxt(os.path.join('/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/',
        'val_samples.txt'))

    # Create features for SNPs:
    X_train = X.loc[X.index.isin(train_idxs)]
    X_test = X.loc[X.index.isin(val_idxs)]
    X_train.to_csv(os.path.join(out_path,'environment_train.csv'))
    X_test.to_csv(os.path.join(out_path,'environment_val.csv'))

def convert_to_csv(file):
      data = file['genotype/matrix']
      df = pd.DataFrame(data[:, :])
      id_list = get_id(file)
      df['id'] = id_list[0]
      return df, id_list

def get_id(file):
      id_sample_hf = file['sample_info/iid']
      id_sample = pd.DataFrame(np.array(id_sample_hf))
      id_sample = id_sample.astype(int)
      return id_sample

def generate_df_for_snps(df_snp, id_df):
      df_height_pred = pd.DataFrame()
      df_height_pred['id'] = id_df[0]
      df_height_pred = pd.merge(df_height_pred, df_snp)
      return df_height_pred

def create_data_hdf5(num_snps,phase,hdf5_path,csv_path):
      hf = h5.File(hdf5_path + f'all.onlyrsids.imputed.noduplicates.recoded.bothsex.{phase}.bellot.top{num_snps}k.hdf5', 'r')
      df_snp, id_df = convert_to_csv(hf)
      hf.close()

      dataset = generate_df_for_snps(df_snp, id_df)

      df = pd.read_csv(csv_path + f'environment_{phase}.csv')
      if 'eid' in df.columns:
          df = df.rename(columns={'eid':'id'})

      # we checked this when creating the environment, but just to be sure
      assert df.isna().any().any() == False

      id_set = set(id_df[0].values)
      environment_ids = set(df['id'].values)
      ids = np.array(list(id_set.intersection(environment_ids)))

      X = dataset.loc[dataset['id'].isin(ids)]
      X = X.sort_values(by=['id'])
      df = df.sort_values(by=['id'])

      assert (X.id.values == df.id.values).all()

      idx = X.id
      X = X.drop(['id'], axis=1)
      df = df.drop(['id'],axis=1)

      with h5.File(data_path + f"Xy_{phase}_{num_snps}k_pca_env.hdf5",'w-') as out_hf:
          X_grp = out_hf.create_group("X")
          y_grp = out_hf.create_group("y")
          X_grp.create_dataset("ID",data=idx.values)
          X_grp.create_dataset("SNP",data=X.values)
          X_grp.create_dataset("Environment",data=df.drop(['50-0.0'],axis=1).values)
          y_grp.create_dataset("Height",data=df['50-0.0'].values)

def prepare_data(num_snps,data_path,hdf5_path,csv_path):
      Xy_train = data_path+f"Xy_train_{num_snps}k_environment.hdf5"
      Xy_val = data_path+f"Xy_val_{num_snps}k_environment.hdf5"
      if os.path.exists(Xy_train) and os.path.exists(Xy_val):
          pass
      else:
          for phase in ['train','val']:
              logging.info(f'Preparing {phase}')
              create_data_hdf5(num_snps,phase,hdf5_path,csv_path)
              logging.info(f'Finished {phase}')

if __name__=='__main__':
      data_path = '/links/groups/borgwardt/Projects/UKBiobank/height_prediction_2021/data/pca_env/'
      hdf5_path = '/links/groups/borgwardt/Data/UKBiobank/genotype_500k/EGAD00010001497/hdf5/'
      csv_path = data_path
      num_snps=10
      create_environment_data(data_path,out_path=data_path)
      prepare_data(num_snps,data_path,hdf5_path,csv_path)
