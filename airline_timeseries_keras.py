# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:47:45 2018

@author: Abhishek S
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.models import Input
from keras.models import Model
from keras.layers import Dense,LSTM,Activation,Dropout,Reshape
from keras import backend as k
from sklearn.preprocessing import StandardScaler




os.chdir('E:/Python')

data=pd.read_csv('airline_timeseries.csv')
data.head()
data.dropna(axis=0,inplace=True)
lookback=2
def data_arrange(data,lookback):
    series=data.iloc[:,1].values
    x_train=np.zeros((len(data)-lookback,lookback))
    for i in range(lookback):
        x_train[:,i]=series[i:len(series)-lookback+i]
    x_train=x_train.reshape(-1,lookback)
    y_train=series[lookback:].reshape(-1,1)
    return x_train,y_train

    
x_tr,y_tr=data_arrange(data,lookback)    
train_x,test_x,train_y,test_y=train_test_split(x_tr,y_tr,test_size=0.3,random_state=154)

std=StandardScaler()
std.fit(x_tr)
x_trs=std.transform(x_tr)

def r_square(y_true,y_pred):
    r2=np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true,axis=0))**2)
    return 1-r2


k.clear_session()
passenger=Input(shape=(lookback,),dtype='float32')
X=Reshape((lookback,1))(passenger)
X=LSTM(8,return_sequences=True)(X)
X=LSTM(8,return_sequences=False)(X)
X=Dense(1)(X)
X=Activation('linear')(X)
model=Model(inputs=passenger,outputs=X)
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_trs,y_tr,epochs=1000,batch_size=1)
y_pred=model.predict(x_trs)
y_pred

r_square(y_tr,y_pred)


y_pred
y_tr
plt.plot(np.arange(0,len(y_tr)),y_tr,color='b')
plt.plot(y_pred)
plt.show()



import matplotlib.pyplot as plt
plt.scatter(y_tr,y_pred)
plt.show()

