# # Preprocessing the Height Datafield

import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import os

class Height_Statistics():
    def __init__(self):
        self.means = {'male':175.67252960367327,'female':162.49780160116387}
        self.stds = {'male':6.843439632761728,'female':6.303442466347971}
        self.age_reg_coeffs = {'b_1':0.021869934349245713, 'b_0':-42.680572362398614}
    def calculate_height(self,height,sex,age):
        correction_height = age*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0']
        height = height + correction_height
        female_idx = np.where(sex==0)
        male_idx = np.where(sex==1)
        height[female_idx] = height[female_idx]*self.stds['female'] + self.means['female']
        height[male_idx] = height[male_idx]*self.stds['male'] + self.means['male']
        return height

class Height_Statistics_Single():
    def __init__(self):
        self.means = {'male':175.67252960367327,'female':162.49780160116387}
        self.stds = {'male':6.843439632761728,'female':6.303442466347971}
        self.age_reg_coeffs = {'b_1':0.021869934349245713, 'b_0':-42.680572362398614}

    def calculate_height(self,height,sex,age):
        correction_height = age*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0']
        height = height + correction_height
        if sex == 0:
            height = height*self.stds['female'] + self.means['female']
        elif sex == 1:
            height = height*self.stds['male'] + self.means['male']
        return height

class Score():
    def __init__(self,means,stds,coeffs):
        self.means = {'male':means[0],'female':means[1]}
        self.stds = {'male':stds[0],'female':stds[1]}
        self.age_reg_coeffs = {'b_1':coeffs[0], 'b_0':coeffs[1]}

    def calculate_score(self,df):
        male_df = df[df['"31-0.0"']==1].copy()
        female_df = df[df['"31-0.0"']==0].copy()
        male_df.loc[:,'score'] = (male_df['"50-0.0"']-self.means['male'])/self.stds['male']
        female_df.loc[:,'score'] = (female_df['"50-0.0"']-self.means['female'])/self.stds['female']
        df = pd.concat([male_df,female_df],axis=0).sort_values('id')
        df.loc[:,'score'] = df['score']-(df['"34-0.0"']*self.age_reg_coeffs['b_1'] + self.age_reg_coeffs['b_0'])
        return df.loc[:,['id','score']]

def load_datafield(number):
    data_path = os.path.join('..','data')
    raw_df = pd.read_csv(data_path + f'/{number}.csv', engine='python', quoting=csv.QUOTE_NONE)
    data = raw_df.replace('"', '', regex=True)
    data['"eid"'] = data['"eid"'].astype(int)
    data.set_index('"eid"')
    data = data.rename(columns={'"eid"': 'id'})
    subset = data.iloc[:, 0:2]
    print(f'Datafield {number} loaded')
    return subset

def test_single():
    print('Test single patient')
    print('-------------------------------')
    df = pd.read_csv(os.path.join('..','data','height_corrected.csv'))
    sex = load_datafield('ukb29983_31')
    age = load_datafield('ukb29983_34')
    height = load_datafield('ukb29983_50')
    df = df.merge(age,on='id').merge(sex,on='id').merge(height,on='id')
    df = df.mask(df['"31-0.0"']=='').dropna()
    df = df.mask(df['"34-0.0"']=='').dropna()
    df = df.mask(df['"50-0.0"']=='').dropna()
    df = df.mask(df['height']==np.nan).dropna()
    df['"50-0.0"'] = df['"50-0.0"'].astype(float)
    df['"31-0.0"'] = df['"31-0.0"'].astype(int)
    df['"34-0.0"'] = df['"34-0.0"'].astype(int)
    print(df.head())

    id_df_train = pd.read_csv(os.path.join('..','data','train_ids.csv'))
    df_train = df[df['id'].isin(id_df_train['id'])]
    df_test = df[~df['id'].isin(id_df_train['id'])]

    stats = Height_Statistics_Single()
    trues = []
    falses = []
    for i,d in enumerate([df_train,df_test]):
        print('Train') if i==0 else print('Test')
        for i in tqdm(range(d.shape[0])):
            test_idx = i
            test_df = d.iloc[test_idx]
            height_pred = stats.calculate_height(test_df['height'],
                                                 test_df['"31-0.0"'],
                                                 test_df['"34-0.0"'])
            if np.allclose(height_pred,test_df['"50-0.0"']):
                trues.append(True)
            else:
                print(False,'\n',height_pred,'\n',test_df['"50-0.0"'])
                falses.append(test_df['id'])
        print('False reconstructions: ', len(falses))
    if np.array(trues).sum() == df.shape[0]:
        return 0
    else:
        return 1

def test_batches():
    print('Test minibatches')
    print('-------------------------------')
    df = pd.read_csv(os.path.join('..','data','height_corrected.csv'))
    sex = load_datafield('ukb29983_31')
    age = load_datafield('ukb29983_34')
    height = load_datafield('ukb29983_50')
    df = df.merge(age,on='id').merge(sex,on='id').merge(height,on='id')
    df = df.mask(df['"31-0.0"']=='').dropna()
    df = df.mask(df['"34-0.0"']=='').dropna()
    df = df.mask(df['"50-0.0"']=='').dropna()
    df = df.mask(df['height']==np.nan).dropna()
    df['"50-0.0"'] = df['"50-0.0"'].astype(float)
    df['"31-0.0"'] = df['"31-0.0"'].astype(int)
    df['"34-0.0"'] = df['"34-0.0"'].astype(int)
    print(df.head())

    id_df_train = pd.read_csv(os.path.join('..','data','train_ids.csv'))
    df_train = df[df['id'].isin(id_df_train['id'])]
    df_test = df[~df['id'].isin(id_df_train['id'])]

    stats = Height_Statistics()
    trues = []
    falses = []
    for i,d in enumerate([df_train,df_test]):
        print('Train') if i==0 else print('Test')
        for i in tqdm(range(10000)):
            test_idx = np.random.randint(0,high=d.shape[0],size=100)
            test_df = d.iloc[test_idx]
            height_pred = stats.calculate_height(test_df['height'].values,
                                                 test_df['"31-0.0"'].values,
                                                 test_df['"34-0.0"'].values)
            if np.allclose(height_pred,test_df['"50-0.0"'].values):
                trues.append(True)
            else:
                false = np.where(np.abs(height_pred-test_df['"50-0.0"'].values) > 0.2)[0]
                print(False,'\n',height_pred[false],'\n',test_df['"50-0.0"'].values[false])
                falses.append(test_df['id'].iloc[false])
        print('False reconstructions: ', len(falses))
    if np.array(trues).sum() == 20000:
        return 0
    else:
        return 1



sex = load_datafield('ukb29983_31')
age = load_datafield('ukb29983_34')
height = load_datafield('ukb29983_50')

age['"34-0.0"'] = age['"34-0.0"'].astype(int)

id_df_train = pd.read_csv(os.path.join('..','data','train_ids.csv'))
sex_train = sex[sex['id'].isin(id_df_train['id'])]
age_train = age[age['id'].isin(id_df_train['id'])]
height_train = height[height['id'].isin(id_df_train['id'])]


sex.head()

age.head()

height.head()

sex_train.shape
age_train.shape
height_train.shape

pd.qcut(age_train['"34-0.0"'],10)

age_train.hist(column='"34-0.0"',bins=len(age_train['"34-0.0"'].unique()),xrot=90,xlabelsize=10)

age_train.groupby(['"34-0.0"']).agg('count')

male_idx_train = sex_train[sex_train['"31-0.0"'].astype(int)==1].index
female_idx_train = sex_train[sex_train['"31-0.0"'].astype(int)==0].index

height_male_train = height_train.loc[male_idx_train]
height_female_train = height_train.loc[female_idx_train]


height_male_train = height_male_train.mask(height_male_train['"50-0.0"']=='').dropna()
height_female_train = height_female_train.mask(height_female_train['"50-0.0"']=='').dropna()

avg_height_male_train = height_male_train['"50-0.0"'].astype(float).mean()
avg_height_female_train = height_female_train['"50-0.0"'].astype(float).mean()
std_height_male_train = height_male_train['"50-0.0"'].astype(float).std()
std_height_female_train = height_female_train['"50-0.0"'].astype(float).std()

print('Male\t\t\t Female')
print(avg_height_male_train,avg_height_female_train)
print(std_height_male_train,std_height_female_train)


height_male_train['height'] = (height_male_train['"50-0.0"'].astype(float)-avg_height_male_train)/std_height_male_train

height_female_train['height'] = (height_female_train['"50-0.0"'].astype(float)-avg_height_female_train)/std_height_female_train

height_corrected_train = pd.concat([height_male_train,height_female_train],axis=0).sort_values('id')

height_corrected_train.head()

height_corrected_train['id']=height_corrected_train['id'].astype(int)

height_corrected_train['height'].mean()
height_corrected_train['height'].std()

age_height_train = age_train.merge(height_corrected_train,on='id')
age_height_train = age_height_train.mask(age_height_train['"34-0.0"']==np.nan).dropna()
age_height_train = age_height_train.mask(age_height_train['"50-0.0"']=='').dropna()
age_height_train.head()

# Only use birth years with at least 3000 patients
important_idx_train = age_height_train[(age_height_train['"34-0.0"'] > 1938) & (age_height_train['"34-0.0"'] <= 1968)].index
X = age_height_train.loc[important_idx_train,'"34-0.0"'].values
X = np.expand_dims(X,axis=1)
y = age_height_train.loc[important_idx_train,'height'].values

model = LinearRegression()
model.fit(X,y)
print(model.coef_[0], model.intercept_)

df = height.merge(age,on='id').merge(sex,on='id')
df = df.mask(df['"31-0.0"']=='').dropna()
df = df.mask(df['"34-0.0"']==np.nan).dropna()
df = df.mask(df['"50-0.0"']=='').dropna()
df['"31-0.0"'] = df['"31-0.0"'].astype(int)
df['"50-0.0"'] = df['"50-0.0"'].astype(float)

scorer = Score(means=[avg_height_male_train,avg_height_female_train],
               stds=[std_height_male_train,std_height_female_train],
               coeffs=[model.coef_[0],model.intercept_])
height_corrected = scorer.calculate_score(df)

height_corrected = height_corrected.rename(columns={'score':'height'})
height_corrected['id'] = height_corrected['id'].astype(int)
height_corrected.head()

data_path = os.path.join('..','data')
height_corrected.to_csv(os.path.join(data_path,'height_corrected.csv'),index=False)

test_single()

test_batches()
