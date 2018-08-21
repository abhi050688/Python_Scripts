# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:09:33 2018

@author: Abhishek S
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

os.chdir("E:/Analytics Vidya/Loan Prediction")

train=pd.read_csv("train.csv")
#train.head()
a=train.groupby('Loan_ID')['Loan_ID'].count()
a[a>1].all()
target='Loan_Status'
X=train.copy()
Y=train[target]
del X[target]
Y=Y.map({'Y':1,'N':0})
np.random.seed(1001)
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2,random_state=100)
float((train_y.sum()))/train_y.count()
float(test_y.sum())/test_y.count()


train_x.head()
train_y.index=train_x['Loan_ID']
train_y.head()
train_x.set_index('Loan_ID',inplace=True)
train_x.info()
train_x.describe()
train_x.dtypes
#Proportion  by gender
b=train_x.groupby(['Gender',train_y])['Gender'].count()
b.groupby(level=0).transform(prop)
b.div(b.groupby(level=0).sum(),level=0)
def prop(series):
    return series/series.sum()
train_x.loc[train_x.Gender.isnull(),'Gender']='Male'
train_x.groupby('Gender')['Gender'].count()
#Fill missing test.Gender with 'Male'
b=test_x.copy()
b.head(20)
test_x.fillna({'Gender':'Male'},inplace=True)

train_x.groupby('Married')['Married'].count()
#Fill Married with Yes

train_x.groupby('Dependents')['Dependents'].count()
train_x.Dependents[train_x.Dependents=='3+']='3'
train_x.Dependents=train_x.Dependents.convert_objects(convert_numeric=True)
test_x.Dependents[test_x.Dependents=='3+']='3'
test_x.Dependents=test_x.Dependents.convert_objects(convert_numeric=True)

#Fill Dependents with 0.0

train_x.groupby('Self_Employed')['Self_Employed'].count()
#Fill Self employed with No
#Fill Loan Amount and Loan Amount term with mean
la_mean=train_x.LoanAmount.mean()
lat_mean=train_x.Loan_Amount_Term.mean()

#Fill Credit History with 1
miss_dict={'Married':'Yes','Dependents':0.0,'Self_Employed':'No','Credit_History':1,'LoanAmount':la_mean,'Loan_Amount_Term':lat_mean}
train_x.fillna(miss_dict,inplace=True)
test_x.fillna(miss_dict,inplace=True)
test_x.info()
test_y.index=test_x.Loan_ID
test_x.set_index('Loan_ID',inplace=True)
cat=list(train_x.columns[train_x.dtypes==np.object])
train_cat=train_x[cat]
train_cat.head()
train_x=pd.get_dummies(train_x,columns=cat,prefix=cat,prefix_sep='_')
test_x=pd.get_dummies(test_x,columns=cat,prefix=cat,prefix_sep='_')

train_x.info()
column_to_drop=['Gender_Female','Married_No','Education_Graduate','Self_Employed_No','Property_Area_Rural']
train_x.drop(column_to_drop,axis=1,inplace=True)
test_x.drop(column_to_drop,axis=1,inplace=True)

regr=LogisticRegression(max_iter=10000)
regr.fit(train_x,train_y)
regr.score(train_x,train_y)
regr.score(test_x,test_y)
regr.predict_proba(train_x)
train_y.groupby(train_y).count()/train_y.count()
r2_score(train_y,regr.predict_proba(train_x)[:,1])
