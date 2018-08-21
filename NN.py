# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 11:01:50 2018

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import normalize

a=np.arange(0,9).reshape(3,3)
normalize(a,norm='l1',axis=0)


def initialize_parameters(nx,ny,activation_list,hidden_layer_dims=list()):
    parameters=dict()
    hidden_layer_dims=[nx]+hidden_layer_dims+[ny]
    for i in xrange(1,len(hidden_layer_dims)):
        if(activation_list[i-1]=='relu' or activation_list[i-1]=='sigmoid'):
            factor=np.sqrt(2./hidden_layer_dims[i-1])
        else:
            factor=np.sqrt(1./hidden_layer_dims[i-1])
        parameters['W'+str(i)]=np.random.randn(hidden_layer_dims[i],hidden_layer_dims[i-1])*factor
        parameters['b'+str(i)]=np.zeros([hidden_layer_dims[i],1])
    return parameters

#para=initialize_parameters(3,1,seed=10)
#for j in xrange(len(para)):
#    print("Para keys= %s with shape %s"%(para.keys()[j],para.get(para.keys()[j]).shape))

def sigmoid(Z):
    return 1/(1+np.exp(-1*Z))
def relu(Z):
    A=np.multiply(Z,Z>=0)
    return(A)
def tanh(Z):
    A=(np.exp(Z) -np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    return(A)

def softmax(Z):
    Z=np.exp(Z)
    A=normalize(Z,axis=0,norm='l1')
    return A

def forward_prop(A_prev,W,b,activation='sigmoid'):
    Z=np.dot(W,A_prev)+b
    if(activation=='sigmoid'):
        A=sigmoid(Z)
    elif(activation=='relu'):
        A=relu(Z)
    elif(activation=='tanh'):
        A=tanh(Z)
    elif(activation=='softmax'):
        A=softmax(Z)
    cache=(W,b,A_prev,Z,activation)
    return A,cache

        
def compute_cost(AL,Y,parameter,l2_norm):
    ny,m=Y.shape
    l=len(parameter)/2
    wt=0
    for i in range(l):
        wt+=np.sum(np.power(parameter['W'+str(i+1)],2))
    if(ny>1):
        cost=np.sum(-np.multiply(Y,np.log(AL)))/m +l2_norm*wt/(2*m)
    elif(ny==1):
        cost=-1*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m +(l2_norm*wt)/(2*m) 
    return cost


def forward_propagation_(X,hidden_layer_dims,parameter,keep_prob,activation='sigmoid'):
    L=len(parameter)/2
    if(type(activation)==str):
        activation=np.repeat(activation,L)
    else:
        activation=np.array(activation)
    A_prev=X
    linear_cache=list()
    for l in range(L):
        if(keep_prob[l]<1):
            A_prev,cache=forward_prop_with_Do(A_prev,parameter["W"+str(l+1)],parameter["b"+str(l+1)],keep_prob[l],activation[l])
        elif(keep_prob[l]==1):
            A_prev,cache=forward_prop(A_prev,parameter["W"+str(l+1)],parameter["b"+str(l+1)],activation[l])
        linear_cache.append(cache)
    return A_prev,tuple(linear_cache)


def dsigmoid(Z):
    A=sigmoid(Z)
    return np.multiply(A,(1-A))
def dtanh(Z):
    A=tanh(Z)
    return 1-np.multiply(A,A)
def drelu(Z):
    return Z>0

def backward_prop(dA,cache):
    m=dA.shape[1]
    W,b,A_prev,Z,activation=cache
    if(activation=='sigmoid'):
        gprime=dsigmoid(Z)
    elif(activation=='tanh'):
        gprime=dtanh(Z)
    elif(activation=='relu'):
        gprime=drelu(Z)
    dZ=np.multiply(dA,gprime)
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)
    return dW,db,dA_prev,dZ
    


def backward_propagation_(Y,AL,linear_cache,keep_prob):
    L=len(linear_cache)
    grads={}
    dA_prev=-np.divide(Y,AL) + np.divide((1-Y),(1-AL))
    for l in range(L-1,-1,-1):
        cache=linear_cache[l]
        if(keep_prob[l]==1):
            grads["dW"+str(l+1)],grads["db"+str(l+1)],dA_prev,grads["dZ"+str(l+1)]=backward_prop(dA_prev,cache)
        elif(keep_prob[l]<1):
            grads["dW"+str(l+1)],grads["db"+str(l+1)],dA_prev,grads["dZ"+str(l+1)]=backward_prop_with_Do(dA_prev,cache,keep_prob[l])
    return grads

def update_parameter(parameter,grads,learning_rate,l2_norm,m):
    l=len(parameter)/2
    for i in xrange(l):
        parameter['W'+str(i+1)]=parameter['W'+str(i+1)]*(1-l2_norm*learning_rate/m)-learning_rate*grads['dW'+str(i+1)]
        parameter['b'+str(i+1)]=parameter['b'+str(i+1)]-learning_rate*grads['db'+str(i+1)]
    return parameter

        
output,cache,costs=neural_network(X,Y,hidden_layer_list,activation_list,learning_rate=5e-2,max_iter=100000,verbose=True)
    

def neural_network(X,Y,hidden_layer_list,activation_list,learning_rate=0.001,keep_prob=1,max_iter=1000,seed=1,l2_norm=0,verbose=False,iter_to_print=100.):
    nx=X.shape[0]
    ny=Y.shape[0]
    m=X.shape[1]
    np.random.seed(seed)
    if(type(keep_prob)==int):
        keep_prob=np.repeat(keep_prob,len(hidden_layer_list)+1)
    keep_prob[len(keep_prob)-1]=1
    assert(len(keep_prob)==len(hidden_layer_list)+1)
    parameter=initialize_parameters(nx,ny,activation_list,hidden_layer_list)
    cost_series=list()
    try:
        for i in xrange(max_iter):
            AL,linear_cache=forward_propagation_(X,hidden_layer_list,parameter,keep_prob,activation_list)
            cost=compute_cost(AL,Y,parameter,l2_norm)
            if math.isnan(cost):
                break
            grads=backward_propagation_(Y,AL,linear_cache,keep_prob)
            parameter=update_parameter(parameter,grads,learning_rate,l2_norm,m)
            cost_series.append(cost)
            if verbose:
                if(i%iter_to_print==0):
                    print('Iteration: %s Cost = %.5f'%(i,cost))
            cache=(parameter,hidden_layer_list,activation_list)
    except KeyboardInterrupt:
        pass
    return AL,cache,cost_series

def nn_predict(cache,X):
    parameter,hidden_layer_list,activation_list=cache
    return forward_propagation_(X,hidden_layer_list,parameter,activation_list)

def decision_boundary(X,Y,cache):
    h=0.2
    x_min,x_max=min(X[0,:])-0.5,max(X[0,:])+0.5
    y_min,y_max=min(X[1,:])-0.5,max(X[1,:])+0.5
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    Z=np.append(xx.ravel().reshape(1,-1),yy.ravel().reshape(1,-1),axis=0)
    A,_=nn_predict(cache,Z)
    A=(A>=0.5)
    A=A.reshape(xx.shape)
    plt.contourf(xx,yy,A)
    plt.scatter(X[0,:],X[1,:],c=Y.flatten(),edgecolor='b')
    plt.show()


def forward_prop_with_Do(A_prev,W,b,keep_prob,activation='sigmoid'):
    Z=np.dot(W,A_prev)+b
    if(activation=='relu'):
        A=relu(Z)
    elif(activation=='tanh'):
        A=tanh(Z)
    elif(activation=='sigmoid'):
        A=sigmoid(Z)
    elif(activation=='softmax'):
        A=softmax(Z)
    D=np.random.rand(A.shape[0],A.shape[1])
    D=(D<=keep_prob)
    A=np.multiply(A,D)
    cache=(W,b,A_prev,Z,D,activation)
    A=A/keep_prob
    return A,cache

def backward_prop_with_Do(dA,cache,keep_prob):
    m=dA.shape[1]
    W,b,A_prev,Z,D,activation=cache
    dA=np.multiply(dA,D)
    dA=dA/keep_prob
    if(activation=='sigmoid'):
        gprime=dsigmoid(Z)
    elif(activation=='relu'):
        gprime=drelu(Z)
    elif(activation=='tanh'):
        gprime=dtanh(Z)
    dZ=np.multiply(dA,gprime)
    dA_prev=np.dot(W.T,dZ)
    dW=np.dot(dZ,A_prev)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    return dW,db,dA_prev,dZ


def backward_propagation_both(Y,AL,linear_cache,keep_prob):
    L=len(linear_cache)
    m=AL.shape[1]
    cache=linear_cache[L-1]
    grads={}
    W,b,A_prev,Z,activation=cache
    dZ=AL-Y
    grads["W"+str(L)]=np.dot(dZ,A_prev.T)/m
    grads["b"+str(L)]=np.sum(dZ,axis=1,keepdims=True)
    dA_prev=np.dot(W.T,dZ)
    for i in range(L-1,0,-1):
        cache=linear_cache[i]
        if(keep_prob==1):
            grads["dW"+str(i)],grads["db"+str(i)],dA_prev,grads["dZ"+str(i)]=backward_prop(dA_prev,cache)
        elif(keep_prob<1):
            grads["dW"+str(i)],grads["db"+str(i)],dA_prev,grads["dZ"+str(i)]=backward_prop_with_Do(dA_prev,cache,keep_prob)
    return grads

    
def compute_cost_softmax(AL,Y,parameter,l2_norm):
    m=AL.shape[1]
    L=len(parameter)/2
    wt=0
    for i in range(L):
        wt+=np.sum(parameter["W"+str(i+1)]**2)
    return total_cost


    
    



