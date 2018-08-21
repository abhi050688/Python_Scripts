# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:47:54 2018

@author: Abhishek S
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Model
from keras.layers import Conv2D,Dense,Dropout,Activation,MaxPool2D,Input,Reshape
from keras.layers import Flatten
import keras.backend as k
from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical

train,test=mnist.load_data()
(tr_x,tr_y),(test_x,test_y)=train,test








#mnist=input_data.read_data_sets("E:/Tensorflow/mnist",one_hot=True)
k.clear_session()
image=Input(shape=(28,28),dtype='float32',name='image')
X=Reshape((1,28,28))(image)
X=Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same')(X)
X=MaxPool2D(pool_size=(2,2),padding='same')(X)
X=Dropout(0.1)(X)
X=Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same')(X)
X=MaxPool2D(pool_size=(2,2),padding='same')(X)
X=Flatten()(X)
X=Dense(200)(X)
X=Activation('relu')(X)
X=Dense(10,activation='sigmoid')(X)
model=Model(inputs=image,outputs=X)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.summary()

tr_x=np.asarray(tr_x,dtype='float32').reshape(-1,28,28)
ts_x=np.asarray(test_x,dtype='float32').reshape(-1,28,28)
tr_y=to_categorical(np.asarray(tr_y,dtype='float32'),num_classes=10)
test_y=to_categorical(np.asarray(test_y,dtype='float32'),num_classes=10)
tr_x=tr_x/255
ts_x=ts_x/255

#for i in range(1000):
#    x,y=
#    model.fit(x=x,y=y,epochs=1)

model.fit(x=tr_x,y=tr_y,epochs=20,batch_size=100,validation_data=(ts_x,test_y))
model.evaluate(ts_x,test_y)

