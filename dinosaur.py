# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 23:56:14 2018

@author: Abhishek S
"""
import numpy as np
import tensorflow.contrib.keras as k
from random import shuffle


file="E:/Deep Learning/Sequence Models/Week 1/Dinosorus/dinos.txt"

with open(file) as f:
    example=[line.lower() for line in f.readlines()]
vocab=list(set("".join(example)))
len("".join(example))

char_to_idx={j:i  for i,j  in enumerate(sorted(vocab))}
idx_to_char={ value:key for key,value in char_to_idx.items()}
#data = open(file, 'r').read()
#data= data.lower()
#chars = list(set(data))
#data_size, vocab_size = len(data), len(chars)
#print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
#char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
#ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
#print(ix_to_char)

def name_to_idx(name):
    a=[ char_to_idx[i] for i in name]
    return a

def idxlist_to_name(idx):
    nm=[idx_to_char[i]  for  i in idx]
    return "".join(nm)
a='abhsgdhd'
b=name_to_idx(a)
print(idxlist_to_name(b))

def softmax(a):
    ez=np.exp((a-np.max(a)))
    return ez/np.sum(ez,axis=0)

#def initialize_parameter(nx,ny,na):
#    parameter={}
#    parameter['Wax'],parameter['Waa'],parameter['Wya'],parameter['b'],parameter['by']=np.random.randn(na,nx)*(1/nx**2),np.random.randn(na,na)/(1/na**2),\
#    np.random.randn(ny,na)*(1/na**2),np.zeros([na,1]),np.zeros([ny,1])
#    return parameter

def initialize_parameter(n_x, n_y,n_a):
    """
    Initialize parameters with small random values
    
    Returns:
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias
    
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    
    return parameters

def rnn_step_forward(xt,parameter,a_prev):
    Wax,Waa,Wya,b,by=parameter['Wax'],parameter['Waa'],parameter['Wya'],parameter['b'],parameter['by']
    a_next=np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+b)
    y=softmax(np.dot(Wya,a_next)+by)
    cache=(y,a_next,a_prev,xt)
    return y.flatten(),a_next,cache



def rnn_forward(X,Y,a_next,parameter):
    Wax,Waa,Wya,b,by=parameter['Wax'],parameter['Waa'],parameter['Wya'],parameter['b'],parameter['by']
    na,nx=Wax.shape
    ny,_=Wya.shape
    caches=[]
    ylist=np.zeros([ny,len(Y)])
    ts=0
    for idx in X:
        xt=np.zeros([ny,1])
        if idx is not None:
            xt[idx]=1
        ylist[:,ts],a_next,cache=rnn_step_forward(xt,parameter,a_next)
        caches.append(cache)
        ts+=1
    return ylist,caches,a_next

def compute_loss(Y,ylist):
    loss=0
    for i in range(len(Y)):
        yi=Y[i]
        loss-=np.log(ylist[yi][i])
    return loss

def update_parameter(parameter,gradient,learning_rate):
    parameter['Wya']-=learning_rate*gradient['dWya']
    parameter['Wax']-=learning_rate*gradient['dWax']
    parameter['Waa']-=learning_rate*gradient['dWaa']
    parameter['b']-=learning_rate*gradient['db']
    parameter['by']-=learning_rate*gradient['dby']
    return parameter

def optimize(X,Y,parameter,a_next,learning_rate):
    ylist,caches,a_next=rnn_forward(X,Y,a_next,parameter)
    loss=compute_loss(Y,ylist)
    gradient=rnn_backward(X,Y,parameter,caches)
    gradient=clip(gradient,5)
    parameter=update_parameter(parameter,gradient,learning_rate)
    return parameter,loss,ylist,a_next
def model(example,learning_rate=.01,vocab_size=27,neurons=50,max_iter=1000,iter_to_print=10):
    na=neurons
    nx,ny=vocab_size,vocab_size
    parameter=initialize_parameter(nx,ny,na)
    costs=[]
    a_next=np.zeros([na,1])
    t_loss=-np.log(1/27)*10
    np.random.seed(1010)
    shuffle(example)
    for  i in range(max_iter):
        idx=i%len(example)
        name=example[idx]
        X=[None]+name_to_idx(name)
        Y=X[1:]+[char_to_idx['\n']]
        parameter,loss,ylist,a_next=optimize(X,Y,parameter,a_next,learning_rate)
        t_loss=.9*t_loss+.1*loss
        costs.append(t_loss)
        if i%iter_to_print==0:
            seed=1
            print("{} loss  epoch == {}".format(loss,i))
            sample(parameter,10,seed)
            seed+=1
    return parameter,costs,ylist




def rnn_step_backward(dy,xt,a_prev,gradient,parameter,a_next):
    Wax,Waa,Wya,b,by=parameter['Wax'],parameter['Waa'],parameter['Wya'],parameter['b'],parameter['by']
    gradient['dWya']+=np.dot(dy,a_next.T)
    gradient['dby']+=dy
    da_next=np.dot(Wya.T,dy)+gradient['da_prev']
    da_next=(1-np.power(a_next,2))*da_next
    gradient['dWax']+=np.dot(da_next,xt.T)
    gradient['dWaa']+=np.dot(da_next,a_prev.T)
    gradient['db']+=da_next
    gradient['da_prev']=np.dot(Waa.T,da_next)
    return gradient

#x.shape=(nx,m,Timestep)
#y.shape=(ny,m,Tx)
def rnn_backward(X,Y,parameter,caches):
    Wax,Waa,Wya,b,by=parameter['Wax'],parameter['Waa'],parameter['Wya'],parameter['b'],parameter['by']
    na,nx=Wax.shape
    ny,_=Wya.shape
    gradient={}
    gradient['dWax'],gradient['dWaa'],gradient['dWya'],gradient['dby'],gradient['db']=np.zeros_like(Wax),np.zeros_like(Waa),np.zeros_like(Wya),np.zeros_like(by),np.zeros_like(b)
    gradient['da_prev']=np.zeros([na,1])
    for i in reversed(range(len(caches))):
        y,a_next,a_prev,xt=caches.pop()
        dy=y
        y[Y[i]]-=1
        gradient=rnn_step_backward(dy,xt,a_prev,gradient,parameter,a_next)
    return gradient


def clip(gradients, maxValue):
    for key,value  in gradients.items():
        gradients[key]=np.clip(gradients[key],-maxValue,maxValue)        
    return gradients
def sample(parameter,names, seed):
    Wax,Waa,Wya,b,by=parameter['Wax'],parameter['Waa'],parameter['Wya'],parameter['b'],parameter['by']
    na,nx=Wax.shape
    counter=0
    eos=char_to_idx['\n']
    for i in range(names):
        indices=[]
        np.random.seed(seed+1)
        indx=np.random.randint(1,nx)
        x=np.zeros([nx,1])
        x[indx,0]=1
        a_prev=np.zeros([na,1])
        while indx!=eos  or counter<50:
            np.random.seed(seed+counter)
            a_prev=np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+b)
            y=softmax(np.dot(Wya,a_prev)+by)
            indx=np.random.choice(np.arange(0,nx),p=y.ravel())
            x=np.zeros([nx,1])
            x[indx,0]=1
            indices.append(indx)
            counter+=1
            seed+=1
        print(idxlist_to_name(indices))
    
parameter_s,costs,ylist=model(example,learning_rate=1e-2,vocab_size=27,neurons=64,max_iter=35000,iter_to_print=1000)