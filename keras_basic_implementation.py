# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 15:58:57 2018

@author: Abhishek S
"""
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import keras as k
from keras.models import Sequential
import keras.layers as l
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.framework import ops

train,labels=make_moons(n_samples=1000,noise=0.4,random_state=1)
train_x,test,label_x,label=train_test_split(train,labels,test_size=0.3,random_state=101)
train_x.shape
test.shape

plt.scatter(train_x[:,0],train_x[:,1],c=label_x.flatten(),edgecolors='b')
plt.show()

x_min

plt.contourf(train_x[:,0],train[:,1],c=label_x)
plt.show()
k.clear_session()
model=Sequential()
model.add(l.Dense(10,activation=l.activations.relu,input_dim=2))
model.add(l.Dropout(.2))
model.add(l.Dense(8,activation=l.activations.tanh))
model.add(l.Dropout(0.2))
model.add(l.Dense(10))
model.add(l.Dropout(.2))
model.add(l.Activation('relu'))
model.add(l.Dense(1,activation=l.activations.sigmoid))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x=train_x,y=label_x,batch_size=64,epochs=1000)
model.evaluate(train_x,label_x)
model.evaluate(test,label)


a,b=np.meshgrid(np.arange(0,9),np.arange(10,18))
a.reshape(-1,1)
b.reshape(-1,1)
a.flatten().shape
a.ravel().shape

