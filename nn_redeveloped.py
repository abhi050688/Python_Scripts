# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 14:41:59 2018

@author: Abhishek S
"""

a=3
a in [1,5]

data=np.random.randn(3,400)
output=np.array(np.random.rand(1,400)>0.5,dtype=int)
layers=[3,4,5,1]
par=initialize(layers)
activations=['relu','tanh','sigmoid']
A,c=forward_propagation(data,par,activations)
grads=backward_propagation(c,output,par,activations.copy())

A.shape
grads=backward_propagation(c,output,par,activations)


p=neural_network(data,layers=[4,5],activations='sigmoid',output_unit=1)
p
