# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:18:11 2018

@author: Abhishek S
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import gc
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import scipy.stats as ss
import matplotlib.pyplot as plt


os.chdir('E:/Analysis/Santandar_Value_Prediction')
os.listdir()
train=pd.read_csv('train.csv')
train.head()
train.info()
train.head(100)
train.describe()
a=train.groupby('ID')['ID'].count()
a[a==1].head()
a.head()
train.target.describe()
np.mean(train.target)
train_x,test_x,train_y,test_y=train_test_split(train.drop(['target','ID'],axis=1),train.target,test_size=0.2,random_state=101)
del train
gc.collect()

train_x


std=MinMaxScaler()
xi=std.fit_transform(train_x)
xi[:,:2]
a=train_x.iloc[:,:2]
b=np.count_nonzero(train_x,axis=0)
np.max(b)
ss.describe(b)
plt.hist(b,bins=np.arange(0,1300,20))
?plt.hist
c=np.sum(train_x,axis=1)
c[c==0]
b[b<100].shape



