# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:03:47 2018

@author: Abhishek S
"""

import numpy as np

a=np.arange(0,4).reshape(-1,2)
b=np.arange(4,8).reshape(-1,2)
np.concatenate([a,b],axis=1)

def softmax(z):
    e_z=np.exp((z-np.max(z)))
    return e_z/np.sum(e_z,axis=0)


def rnn_cell_forward(xt,a_prev,parameter):
    Wax=parameter['Wax']
    Waa=parameter['Waa']
    Wya=parameter['Wya']
    b=parameter['b']
    by=parameter['by']
    Waxa=np.concatenate([Wax,Waa],axis=1)
    X=np.concatenate([xt,a_prev],axis=0)
    A=np.tanh(np.dot(Waxa,X)+b)
    Y=softmax(np.dot(Wya,A)+by)
    cache=(A,a_prev,xt)
    return A,Y,cache

np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
b = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": b, "by": by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", a_next.shape)
print("yt_pred[1] =", yt_pred[1])
print("yt_pred.shape = ", yt_pred.shape)


#x has shape(nx,m,Tx)
#a0 has shape(na,m)
#Wax hahs shape(na,nx)
#Waa has shape(na,na)
#Wya has shape(ny,na)
def rnn_forward(x,a0,parameter):
    nx,m,Tx=x.shape
    ny,na=parameter['Wya'].shape
    caches=list()
    a_series=np.zeros([na,m,Tx])
    a_next=a0
    ylist=np.zeros([ny,m,Tx])
    for i in range(Tx):
        a_next,ylist[:,:,i],cache=rnn_cell_forward(x[:,:,i],a_next,parameter)
        a_series[:,:,i]=a_next
        caches.append(cache)
    return a_series,ylist,caches


np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
b = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "b": b, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y_pred[1][3] =", y_pred[1][3])
print("y_pred.shape = ", y_pred.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches))


def rnn_step_backward(dy,xt,a_prev,da_prev,gradient,parameters,a_next):
    Wax,Waa,Wya,b,by=parameters['Wax'],parameters['Waa'],parameters['Wya'],parameters['b'],parameters['by']
    gradient['dWya']+=np.dot(dy,a_next.T)
    gradient['dby']+=np.sum(dy,axis=1,keepdims=True)
    da_next=np.dot(Wya.T,dy)+da_prev
    gradient['dWax']+=np.dot(da_next,xt.T)
    gradient['dWaa']+=np.dot(da_next,a_prev.T)
    gradient['db']+=np.sum(da_next,axis=1,keepdims=True)
    da_prev=np.dot(Waa.T,gradient['da_next'])
    return gradient,da_prev

#x.shape=(nx,m,Timestep)
#y.shape=(ny,m,Tx)
def rnn_backward(X,Y,ylist,a_series,parameters,caches):
    nx,m,Tx=X.shape
    ny,_=Y.shape
    a,_,_=caches[0]
    na,_=a.shape
    gradient={}
    Wax,Waa,Wya,b,by=parameters['Wax'],parameters['Waa'],parameters['Wya'],parameters['b'],parameters['by']
    gradient['dWax'],gradient['dWaa'],gradient['dWya'],gradient['by'],gradient['b']=np.zeros_like(Wax),np.zeros_like(Waa),np.zeros_like(Wya),np.zeros_like(by),np.zeros_like(b)
    loss=0
    da_prev=np.zeros_like(a)
    for i in reversed(range(Tx)):
        a_next,a_prev,xt=caches.pop()
        dy=ylist[:,:,i]-Y[:,:,i]
        loss-=np.sum(np.multiply(np.log(ylist[:,:,i]),Y[:,:,i]))
        gradient,da_prev=rnn_step_backward(dy,xt,a_prev,da_prev,gradient,parameters,a_next)
    return gradient

        











