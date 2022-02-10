# # Preprocessing the Height Datafield
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def evaluate_metrics(phenotype,predictions):
    return({'MSE':mean_squared_error(phenotype,predictions),
            'MAE':mean_absolute_error(phenotype,predictions),
            'r':np.corrcoef(phenotype,predictions)[0,1],
            'r2':r2_score(phenotype,predictions)})


DATA_PATH = '/home/michael/ETH/data/HeightPrediction/childhood_and_environment'

files = glob(os.path.join(DATA_PATH,'*.csv'))

df_list = [pd.read_csv(f,index_col=0) for f in files]
df_y = pd.read_csv(os.path.join('/home/michael/ETH/data/HeightPrediction/','ukb29983_50.csv'),index_col=0)

df = pd.concat(df_list+[df_y],axis=1)

cols_to_select = df.isna().mean()[df.isna().mean() < 0.11].index

X = df.loc[:,cols_to_select].dropna(axis=0)

train_idxs = np.loadtxt(os.path.join('/home/michael/ETH/data/HeightPrediction/','train_samples.txt'))
val_idxs = np.loadtxt(os.path.join('/home/michael/ETH/data/HeightPrediction/','val_samples.txt'))

X_train = X.loc[X.index.isin(train_idxs),['31-0.0','21022-0.0']]#.drop('50-0.0',axis=1)
X_test = X.loc[X.index.isin(val_idxs),['31-0.0','21022-0.0']]#.drop('50-0.0',axis=1)

y_train = X.loc[X_train.index,'50-0.0']
y_test = X.loc[X_test.index,'50-0.0']

assert y_train.isna().any().any()==False
assert y_test.isna().any().any()==False

# model = LinearRegression()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
model = GradientBoostingRegressor()
model.fit(X_train.values,y_train.values)
predictions=model.predict(X_test.values)
print(evaluate_metrics(y_test.values,predictions))

importances = {X_train.columns[i]:model.feature_importances_[i] for i in range(len(X_train.columns))}

plt.figure()
plt.bar(*zip(*importances.items()))
plt.show()

# Create features for SNPs:
# X_train = X.loc[X.index.isin(train_idxs)]
# X_test = X.loc[X.index.isin(val_idxs)]
#
# X_train.to_csv('environment_train.csv')
# X_test.to_csv('environment_val.csv')
