# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 19:56:24 2018

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

os.chdir('E:/Analytics Vidya/product_purchased')
train=pd.read_csv('train.csv',dtype=dtype_dict)
target='Purchase'
train.info()
x_train,x_test,y_train,y_test=train_test_split(train.drop(columns=['Purchase'],axis=1),train[target],test_size=0.2)
x_train.info()
dtype_dict=dict(zip(x_train.columns,np.repeat(np.object,x_train.shape[1])))
cat=x_train.columns[x_train.dtypes==np.object]
train.head()






