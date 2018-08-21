# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:13:08 2018

@author: Abhishek S
"""
import numpy as np
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(1)
data=np.random.randn(3,400)
output=np.array(np.random.rand(1,400)>0.5,dtype=int)
layers=[2,4,5,1]
activations=['relu','relu','sigmoid']
par=initialize(layers)
A,c=forward_propagation(data,par,activations)

A.shape
len(c)
grads=backward_propagation(c.copy(),output,par,activations.copy())
grads['dW1'].shape

np.sum(data,axis=1).shape

grads

A.shape
grads=backward_propagation(c,output,par,activations)


x_train,y_train=datasets.make_moons(noise=0.8,random_state=1)
std=StandardScaler()
x_train=std.fit_transform(x_train)
x_train=x_train.T
y_train=y_train.reshape(1,-1)
y_train.shape
x_train.shape

plt.scatter(x_train[0,:],x_train[1,:],c=y_train.flatten(),edgecolor='b')

para=initialize(layers)
a=grad_check(x_train,y_train,activations,para)



A,parameter,costs=neural_network(x_train,y_train,[4,5,5],['relu','relu','tanh','sigmoid'],learning_rate=1e-2,output_unit=1,seed=1,max_iter=200000,iter_to_print=1000,verbose=True)
plt.plot(costs)

A,parameter,costs=neural_network(x_train,y_train,layers=[4,5,5],activations=['relu','relu','tanh','sigmoid'],keep_prob=1.0,\
                   learning_rate=1e-2,forb_norm=0,output_unit=1,seed=101,max_iter=100000,\
                   gradient_check=False,grad_check_iter=500,iter_to_print=100,verbose=True)
plt.plot(costs)

A_l2,_,costs_l2=neural_network(x_train,y_train,layers=[4,5,5],activations=['relu','relu','tanh','sigmoid'],keep_prob=1.0,\
                   learning_rate=1e-2,forb_norm=5e-2,output_unit=1,seed=101,max_iter=100000,\
                   gradient_check=False,grad_check_iter=500,iter_to_print=100,verbose=False)


A_k,_,costs_k=neural_network(x_train,y_train,layers=[4,5,5],activations=['relu','relu','tanh','sigmoid'],keep_prob=0.95,\
                   learning_rate=1e-2,forb_norm=0,output_unit=1,seed=101,max_iter=100000,\
                   gradient_check=False,grad_check_iter=500,iter_to_print=100,verbose=False)
plt.plot(costs)
plt.plot(costs_l2)
plt.plot(costs_k)
plt.show()




os.chdir('S:/Anaconda/Python_Scripts')




import NN as nn

AL,cache,cost_series=nn.neural_network(x_train,y_train,[2,3],activations,learning_rate=0.001,keep_prob=1,max_iter=1000,seed=1,l2_norm=0,verbose=True,iter_to_print=1.)