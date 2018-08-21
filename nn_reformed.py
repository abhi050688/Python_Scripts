# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 14:47:14 2018

@author: Abhishek S
"""

import numpy as np
import matplotlib.pyplot as plt
k=1.
type(k)==float


def neural_network(data,output,layers=[4,5],activations='sigmoid',keep_prob=1.0,\
                   learning_rate=1e-2,forb_norm=0,output_unit=1,batch_size=None,seed=101,max_iter=1000,\
                   gradient_check=False,grad_check_iter=500,beta1=0.9,beta2=0.99,iter_to_print=100,verbose=False):
    nx,m=data.shape
    layers=[nx]+layers+[output_unit]
    if type(activations)==str:
        activations=list(np.repeat(activations,len(layers)-1))
    else:
        assert(len(layers)-1==len(activations))
    if type(keep_prob)==float:
        keep_prob=list(np.repeat(keep_prob,len(layers)-2))+[1.0]
    else:
        assert(len(keep_prob)==len(layers)-1)
    if batch_size is None:
        batch_size=m
    batch_list=create_batches(m,batch_size)
    momentum=0
    rms=0
    parameters={}
    np.random.seed(seed)
    parameters,momentum,rms=initialize(layers,activations)
    costs=[]
    for i in range(max_iter):
        for e in range(1,len(batch_list)):
            X=data[:,batch_list[e-1]:batch_list[e]]
            Y=output[:,batch_list[e-1]:batch_list[e]]
            A,caches=forward_propagation(X,parameters,activations,keep_prob)
            cost=compute_cost(A,Y,parameters,forb_norm)
            grads=backward_propagation(caches,Y,parameters,activations.copy())
            momentum=compute_momentum(grads,beta1,momentum)
            rms=compute_rms(grads,beta2,rms)
            parameters=update_parameters(momentum,rms,parameters,learning_rate,forb_norm,m,beta1,beta2,i*(len(batch_list)-1)+e)
        if (gradient_check) and (i%grad_check_iter==0):
            diff=grad_check(data,output,activations,parameters)
            print("Gradient Check for epsilon 1e-7 at iter %d = %e"%(i,diff))
        if i%iter_to_print==0:
            costs.append(cost)
            if verbose:
                print("Cost at iteration: %d  =  %.8f"%(i,cost))
    return A,parameters,costs

def compute_momentum(grads,beta1,momentum):
    L=int(len(grads)/2)
    new_moment=dict()
    for i in range(1,L+1):
        new_moment['W'+str(i)]=beta1*momentum['W'+str(i)]+(1-beta1)*grads['W'+str(i)]
        new_moment['W'+str(i)]=beta1*momentum['b'+str(i)]+(1-beta1)(grads['b'+str(i)])
    return new_moment

def  compute_rms(grads,beta2,rms):
    L=int(len(grads)/2)
    new_rms={}
    for i in range(1,L+1):
        new_rms['W'+str(i)]=beta2*rms['W'+str(i)]+(1-beta2)*(np.power(grads['W'+str(i)],2))
        new_rms['b'+str(i)]=beta2*rms['b'+str(i)]+(1-beta2)*(np.power(grads['b'+str(i)],2))
    return new_rms

def create_batches(m,batch_size):
    n=int(m/batch_size)
    rem=m%batch_size
    l=list()
    for i in range(n+1):
        l.append(i*batch_size)
    if rem!=0:
        l.append((i)*batch_size+rem)
    return l

def forward_step(W,b,A_prev,activation,keep_activation):
    Z=np.dot(W,A_prev)+b
    k=np.random.rand(Z.shape[0],Z.shape[1])<=keep_activation
    A=np.divide(np.multiply(compute_Z_to_A(Z,activation),k),keep_activation)
    cache=(W,b,A_prev,Z,A,activation,k,keep_activation)
    return A,cache

def forward_propagation(data,parameters,activations,keep_prob):
    L=int(len(parameters)/2)
    caches=[]
    A_prev=data.copy()
    for j in range(1,L+1):
        A_prev,cache=forward_step(parameters['W'+str(j)],parameters['b'+str(j)],A_prev,activations[j-1],keep_prob[j-1])
        caches.append(cache)
    return A_prev,caches
def sigmoid(Z):
    return 1/(1+np.exp(-Z))
def tanh(Z):
    return np.tanh(Z)
def softmax(Z):
    pass
def drelu(Z):
    return Z>=0
def dsigmoid(Z):
    A=sigmoid(Z)
    return A*(1-A)
def dtanh(Z):
    A=tanh(Z)
    return 1-np.power(A,2)
def relu(Z):
    return Z*(Z>=0)
def backward_propagation(caches,Y,parameters,activations):
    L=len(caches)
    cache=caches.pop()
    W,b,A_prev,Z,A,activation,k,keep_activation=cache
    grads={}
    dZ=A-Y
    grads['dW'+str(L)],grads['db'+str(L)],dA=backward_step(dZ,A_prev,W,b,k,keep_activation)
    for i in range(L-1,0,-1):
        cache=caches.pop()
        W,b,A_prev,Z,A,activation,k,keep_activation=cache
        dZ=gprime(Z,activation)*dA
        grads['dW'+str(i)],grads['db'+str(i)],dA=backward_step(dZ,A_prev,W,b,k,keep_activation)
    return grads
def backward_step(dZ,A_prev,W,b,k,keep_activation):
    m=A_prev.shape[1]
    dZ=np.divide(np.multiply(dZ,k),keep_activation)
    dW=(np.dot(dZ,A_prev.T))/m
    db=(np.sum(dZ,axis=1,keepdims=True))/m
    dA_prev=np.dot(W.T,dZ)
    return dW,db,dA_prev
def compute_Z_to_A(Z,activation):
    if activation=='relu':
        A=relu(Z)
    elif activation=='tanh':
        A=tanh(Z)
    elif activation=='sigmoid':
        A=sigmoid(Z)
    elif activation=='softmax':
        A=softmax(Z)
    elif activation=='linear':
        A=Z
    else:
        raise ValueError('Activation value not found')
    return A
def gprime(Z,activation):
    if activation=='relu':
        gprime=drelu(Z)
    elif activation=='sigmoid':
        gprime=dsigmoid(Z)
    elif activation=='tanh':
        gprime=dtanh(Z)
    elif activation=='linear':
        gprime=1
    return gprime

def update_parameters(momentum,rms,parameters,learning_rate,forb_norm,m,beta1,beta2,iteration):
    L=int(len(parameters)/2)
    eps=1e-7
    m_corr=1/(1-np.power(beta1,iteration))
    r_corr=1/(1-np.power(beta2,iteration))
    for i in range(1,L+1):
        gW=np.divide(momentum['W'+str(i)]*m_corr,np.power(rms['W'+str(i)]*r_corr,0.5)+eps)
        gb=np.divide(momentum['b'+str(i)]*m_corr,np.power(rms['b'+str(i)]*r_corr,0.5)+eps)
        parameters['W'+str(i)]=parameters['W'+str(i)]*(1-forb_norm*learning_rate/m) - learning_rate*gW
        parameters['b'+str(i)]=parameters['b'+str(i)] - learning_rate*gb
    return parameters
def compute_cost(A,output,parameters,forb_norm):
    m=A.shape[1]
    wt=compute_weights(parameters)
    cost=-1/m * np.sum(output*np.log(A)+(1-output)*np.log(1-A))+(forb_norm/(2*m))*wt
    return cost
def compute_weights(parameters):
    L=int(len(parameters)/2)
    wt=0
    for i in range(1,L+1):
        wt+=np.sum(np.power(parameters['W'+str(i)],2))
    return wt
def initialize(layers,activations):
    params={}
    momentum={}
    rms={}
    for i in range(1,len(layers)):
        if activations[i-1] in ['tanh','sigmoid','softmax']:
            params['W'+str(i)]=np.random.randn(layers[i],layers[i-1])*(1/np.power(layers[i-1],0.5))
        else:
            params['W'+str(i)]=np.random.randn(layers[i],layers[i-1])*(2/np.power(layers[i-1],0.5))
        params['b'+str(i)]=np.zeros([layers[i],1])
        momentum['W'+str(i)]=np.zeros([layers[i],layers[i-1]])
        momentum['b'+str(i)]=np.zeros([layers[i],1])
    rms=momentum.copy()
    return params,momentum,rms

def grad_check(data,Y,activations,parameters):
    A,caches=forward_propagation(data,parameters,activations)
    grads=backward_propagation(caches,Y,parameters,activations.copy())
    epsilon=1e-7
    gradient=list()    
    vector=dict_to_vector_p(parameters)
    for i in range(len(vector)):
        vector=dict_to_vector_p(parameters)
        vector[i]+=epsilon
        A,_=forward_propagation(data,vector_to_dict(vector,parameters),activations)
        J_plus=compute_cost(A,Y)
        vector[i]-=2*epsilon
        A,_=forward_propagation(data,vector_to_dict(vector,parameters),activations)
        J_minus=compute_cost(A,Y)
        gradient.append((J_plus-J_minus)/(2*epsilon))
    grads_v=np.array(dict_to_vector_g(grads))
    gradient=np.array(gradient)
    num=np.power(np.dot((grads_v-gradient).T,(grads_v-gradient)),0.5)
    den=np.power(np.dot(grads_v.T,grads_v),0.5)+np.power(np.dot(gradient.T,gradient),0.5)
    difference=num/den
    return difference

def dict_to_vector_p(parameters):
    L=int(len(parameters)/2)
    vector=[]
    for i in range(1,L+1):
        vector+=list(parameters['W'+str(i)].flatten())
        vector+=list(parameters['b'+str(i)].flatten())
    return vector

def dict_to_vector_g(parameters):
    L=int(len(parameters)/2)
    vector=[]
    for i in range(1,L+1):
        vector+=list(parameters['dW'+str(i)].flatten())
        vector+=list(parameters['db'+str(i)].flatten())
    return vector

def vector_to_dict(vector,parameters):
    L=int(len(parameters)/2)
    new_par=dict()
    start=0
    for i in range(1,L+1):
        n1,n2=parameters['W'+str(i)].shape
        new_par['W'+str(i)]=np.array(vector[start:start+n1*n2]).reshape(n1,n2)
        start+=n1*n2
        n3,_=parameters['b'+str(i)].shape
        new_par['b'+str(i)]=np.array(vector[start:start+n3]).reshape(n3,1)
        start+=n3
    return new_par


A_k,_,costs_k=neural_network(x_train,y_train,layers=[4,5,5],activations=['relu','relu','tanh','sigmoid'],keep_prob=0.95,\
                   learning_rate=1e-2,forb_norm=0,output_unit=1,seed=101,max_iter=100000,\
                   gradient_check=False,grad_check_iter=500,iter_to_print=100,verbose=False)







