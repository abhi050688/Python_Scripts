# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:39:53 2018

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

os.chdir("E:\Analytics Vidya\Loan Prediction")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
train.head()
train.info()
Y=train[target]
X=train.copy()
del X[target]

target='Loan_Status'
missing_loan=train.loc[train.LoanAmount.isna(),[target]]
missing_loan.groupby(target)[target].count()
train.describe()
np.random.seed(1001)
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2)
train_x.head()
train_x.describe()
train_x.groupby('Credit_History',train)[target].count()
train_x.loc[train_x.Credit_History.isna(),'Credit_History']=1.0
mn_LA=train_x.LoanAmount.mean()
mn_LAT=train_x.Loan_Amount_Term.mean()
train_x.loc[train_x.LoanAmount.isna(),'LoanAmount']=mn_LA
train_x.loc[train_x.Loan_Amount_Term.isna(),'Loan_Amount_Term']=mn_LAT
test_x.loc[test_x.Credit_History.isna(),"Credit_History"]=1.0
test_x.loc[test_x.LoanAmount.isna(),'LoanAmount']=mn_LA
test_x.loc[test_x.Loan_Amount_Term.isna(),'Loan_Amount_Term']=mn_LAT
train_x.describe()
train_x.info()
train_x['Dependents'].unique()
train_x.loc[train_x.Dependents=='3+','Dependents']=3
train_x['Dependents']=train_x['Dependents'].convert_objects(convert_numeric=True)
train_x.groupby('Dependents')['Dependents'].count()
train_x['Dependents'].fillna(value=0.0,inplace=True)
dict_fill={'Gender':'NA','Married':'NA','Self_Employed':'NA'}
train_x.fillna(value={'Gender':'NA','Married':'NA','Self_Employed':'NA'},inplace=True)

data=train_x.copy()
del data['Loan_ID']
data.head()

def df_one_hot_encoding(data):
    category=list(data.columns[data.dtypes==np.object])
    for i in range(len(category)):
        clm=category[i]
        temp=pd.get_dummies(data[clm],drop_first=True,prefix=clm)
        data=pd.concat([data,temp],axis=1)
        del data[clm]
    return data
dt=df_one_hot_encoding(data)
dt.head()
train_y=train_y.map({'Y':1,'N':0})
train_y.head()

regr=LogisticRegression(max_iter=10000)
regr.fit(dt,train_y)
regr.coef_
regr.score(dt,train_y)

test_x.info()
test_x.fillna(value=dict_fill,inplace=True)
test_x.loc[test_x.Dependents=='3+','Dependents']=3
test_x['Dependents'].fillna(value=0.0,inplace=True)
test_x['Dependents']=test_x['Dependents'].convert_objects(convert_numeric=True)
data_test=test_x.copy()
del data_test['Loan_ID']
dt_test=df_one_hot_encoding(data_test)
dt_test.head()
test_y=test_y.map({'Y':1,'N':0})
regr.score(dt_test,test_y)
