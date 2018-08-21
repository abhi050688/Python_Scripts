# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 23:12:43 2018

@author: Abhishek S
"""
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import gc

#==================================================================================================================#
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import cpu_count, pool
import pickle



pd.options.mode.chained_assignment=None
train=pd.read_csv(r"E:\Analysis\Movie-setiment\train.tsv",delimiter='\t')
#a=train.loc[train.SentenceId.isin(list(range(1,12))),:]
a=train.copy()
a=a.loc[a.Phrase!=' ',:]

a['beg']=a['Phrase'].apply(lambda x: x.split()[0])
a['end']=a['Phrase'].apply(lambda x: x.split()[len(x.split())-1])

class tree_creation:
    def __init__(self,data):
        self.a=data
        self.beg=self.a.groupby('beg').groups
        self.end=self.a.groupby('end').groups
        self.a['trees']=np.repeat(dict(),self.a.shape[0])
        
    def create_tree(self,row):
        tree=dict()
        line=row.Phrase.iat[0]
        if len(line.split())==1:
            return self.create_leaf(row)
        else:
            candidates1=self.beg[row.beg.iat[0]]
            candidates2=self.end[row.end.iat[0]]
            for j in candidates1:
                for k in candidates2:
                    if self.a.Phrase[j]+" "+self.a.Phrase[k]==line:
                        tree['left']=self.create_tree(self.a.loc[self.a.index==j,:])
                        tree['right']=self.create_tree(self.a.loc[self.a.index==k,:])
                        tree['left_word']=self.a.Phrase[j]
                        tree['right_word']=self.a.Phrase[k]
                        tree['leaf']=False
                        tree['word']=line
                        return tree
            return self.create_leaf(row)
    def create_leaf(self,row):
        line=row.Phrase
        tree={}
        tree['leaf']=True
        tree['word']=line
        tree['idx']=row.PhraseId
        return tree
    
    def run(self):
        for i in self.a.index:
            row=a.loc[a.index==i,:]
            self.a.at[i,'trees']=self.create_tree(row)
        return self.a
    

def generate_tree(data):
    new_tree=tree_creation(data)
    tree=new_tree.run()
    del new_tree
    gc.collect()
    return tree
train=pd.read_csv(r"E:\Analysis\Movie-setiment\train.tsv",delimiter='\t')
cores=cpu_count()
partitions=cores

def parallelize(data,func):
    data_split=np.array_split(data,partitions)
    pl=pool.Pool(cores)
    data=pd.concat(pl.map(func,data_split))
    pl.close()
    pl.join()
    return data

final=parallelize(a,generate_tree)

final=pd.DataFrame()
nsentence=train.SentenceId.nunique()
rngs=list(range(0,nsentence,100))+[nsentence]
for i in range(len(rngs)-1):
    print(i)
    a=train.loc[train.SentenceId.isin(list(range(rngs[i],rngs[i+1]))),:]
    a=a.loc[a.Phrase!=' ',:]
    a['beg']=a['Phrase'].apply(lambda x: x.split()[0])
    a['end']=a['Phrase'].apply(lambda x: x.split()[len(x.split())-1])
    new_data=a.groupby('SentenceId').apply(generate_tree)
    final=pd.concat([final,new_data])

with open(r"E:\Analysis\Movie-setiment\trees.obj" , 'wb') as wf:
    pickle.dump(final,wf)


new_data.shape

new=tree_creation(a)
n=new.run()


#print(a.head())
#print(beg)
n_a.iloc[55,6]
a.head()
a['trees']=np.repeat(dict(),a.shape[0])



for i in range(a.shape[0]):
    row=a.iloc[i,:]
    a.set_value(i,'trees',create_tree(row))

def create_stump(b):
    print("--------{}--------".format(b['word']))
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    if not b['leaf']:
        print("{}--------{}".format(b['left']['word'],b['right']['word']))
    
#create_stump(b)
#create_stump(b['right'])
#create_stump(b['left'])
#create_stump(b['right']['right'])
#create_stump(b['right']['right']['right']['right']['right'])

#Wa=np.random.randn(5,50)*(2/50**2)
#ba=np.zeros([5,1])
#z=np.dot(Wa,b['embedding'])+ba
#ahat=softmax(z)
#y=np.array([1,0,0,0,0],dtype='float').reshape(-1,1)
#dz=ahat-y
#gradients=dict()
#gradients['dWa']=np.dot(dz,b['embedding'].T).shape
#gradients['dba']=dz
#da_prev=np.dot(Wa.T,dz)
#da_prev.shape
#a_prev=np.concatenate([b['left']['embedding'],b['right']['embedding']],axis=0)
#gprime=dtanh(a_prev)
#dz=da_prev*gprime
#gradients['dWy']=np.dot(dz,a_prev.T)
#gradients['dWy'].shape
#da_prev_combined=np.dot(Wy.T,dz)


def softmax(z):
    ez=np.exp(z-max(z))
    return ez/np.sum(ez,axis=0)
def dtanh(a):
    return (1-np.power(a,2))

def initalize_parameters(embedding_size,ny,seed=101):
    parameters=dict()
    np.random.seed(seed)
    parameters['Wa']=np.random.randn(ny,embedding_size)*(1/(ny**2))
    parameters['Wy']=np.random.randn(embedding_size,2*embedding_size)*(2/(embedding_size**2))
    parameters['ba']=np.zeros([ny,1])
    parameters['by']=np.zeros([embedding_size,1])
    return parameters
def initialize_gradients(parameters):
    gradients=dict()
    gradients['dWa']=np.zeros_like(parameters['Wa'])
    gradients['dWy']=np.zeros_like(parameters['Wy'])
    gradients['dba']=np.zeros_like(parameters['ba'])
    gradients['dby']=np.zeros_like(parameters['by'])
    return gradients
def forward_step(tree,parameters):
    Wy,by=parameters['Wy'],parameters['by']
    if tree['leaf']:
        lngth=len(tree['word'].split())
        if lngth>1:
            embed=np.zeros([50,1])
            for i in tree['word'].split():
                embed+=embedding[:,vocab_dict[i]].reshape(-1,1)
            embed=embed/lngth
            tree['embedding']=embed.reshape(-1,1)
        else:
            tree['embedding']=embedding[:,vocab_dict[tree['word']]].reshape(-1,1)
        return tree['embedding']
    else:
        embed=np.zeros([100,1])
        embed[:50,:]=forward_step(tree['left'],parameters)
        embed[50:,:]=forward_step(tree['right'],parameters)
        tree['embedding']=np.dot(Wy,embed)+by
        return tree['embedding']


def backward_propagation(tree,parameters,gradients,y,ahat):
    dz=ahat-y
    a_prev=tree['embedding']
    gradients['dWa']=np.dot(dz,a_prev.T)
    gradients['dba']=dz
    da_prev=np.dot(parameters['Wa'].T,dz)
    backward_step(tree,da_prev,parameters,gradients)
    return gradients



def backward_step(tree,da,parameters,gradients):
    if not tree['leaf']:
        gprime=dtanh(tree['embedding'])
        dz=gprime*da
        a_prev=np.concatenate([tree['left']['embedding'],tree['right']['embedding']],axis=0)
        gradients['dWy']+=np.dot(dz,a_prev.T)
        gradients['dby']+=dz
        da_prev=np.dot(parameters['Wy'].T,dz)
        da_prev_left=da_prev[:embedding_size,:]
        da_prev_right=da_prev[embedding_size:,:]
        backward_step(tree['left'],da_prev_left,parameters,gradients)
        backward_step(tree['right'],da_prev_right,parameters,gradients)
def forward_propagation(parameters,tree):
    Wa,ba=parameters['Wa'],parameters['ba']
    _ =forward_step(tree,parameters)
    z=np.dot(Wa,tree['embedding'])+ba
    ahat=softmax(z)
    return ahat
def update_parameters(momentum,rms,parameters,learning_rate,iteration,l2_norm,epsilon):
    parameters['Wy']=parameters['Wy']*(1-l2_norm*learning_rate)-learning_rate * momentum['dWy']/(1-beta1**iteration) * (1-beta2**2)/(np.power(rms['dWy'],0.5)+epsilon)
    parameters['by']=parameters['by']*(1-l2_norm*learning_rate)-learning_rate * momentum['dby']/(1-beta1**iteration) * (1-beta2**2)/(np.power(rms['dby'],0.5)+epsilon)
    parameters['Wa']=parameters['Wa']*(1-l2_norm*learning_rate)-learning_rate * momentum['dWa']/(1-beta1**iteration) * (1-beta2**2)/(np.power(rms['dWa'],0.5)+epsilon)
    parameters['ba']=parameters['ba']*(1-l2_norm*learning_rate)-learning_rate * momentum['dba']/(1-beta1**iteration) * (1-beta2**2)/(np.power(rms['dba'],0.5)+epsilon)
    return parameters

def update_momentum(momentum,rms,gradients,beta1,beta2):
    for  i in momentum.keys():
        momentum[i]=beta1*momentum[i]+(1-beta1)*gradients[i]
        rms[i]=beta2*rms[i]**2+(1-beta2)*gradients[i]**2
    return momentum,rms
def compute_cost(output,ahat):
    return -np.sum(output*np.log(ahat))
    

i=55
a.iloc[i,2]
b=a.iloc[i,6]
embedding_size=50
beta1=0.9
beta2=0.99
iteration=1
l2_norm=1e-2
learning_rate=1e-2
epsilon=1e-4
trees=a.iloc[:,6].values
labels=a.iloc[:,3]
m_str=set(max(a.Phrase,key=len).split(' '))
vocab_dict={j:i  for i,j in enumerate(m_str)}
embedding=np.random.randn(embedding_size,len(m_str))
def treenet(trees,labels,epochs=100,embedding_size=50,beta1=0.9,beta2=0.999,l2_norm=1e-2,learning_rate=1e-2,epsilon=1e-7,iter_to_print=2):
    assert(type(trees)==np.ndarray)
    m=len(trees)
    parameters=initalize_parameters(50,5,102)
    gradients=initialize_gradients(parameters)
    momentum=gradients.copy()
    rms=gradients.copy()
    enc=OneHotEncoder(5,dtype='float32')
    y=enc.fit_transform(labels.values.reshape(-1,1))
    y=y.toarray().T
    costs=[]
    avg_cost=0
    for epoch in range(epochs):
        for iteration in range(m):
#            print(parameters['Wa'][:1,:])
            output=y[:,iteration].reshape(-1,1)
            tree=trees[iteration]
            ahat=forward_propagation(parameters,tree)
            j=compute_cost(output,ahat)
            gradients=backward_propagation(tree,parameters,gradients,output,ahat)
            momentum,rms=update_momentum(momentum,rms,gradients,beta1,beta2)
            gradients=initialize_gradients(parameters)
            parameters=update_parameters(momentum,rms,parameters,learning_rate,epoch*m +iteration+1,l2_norm,epsilon)
            costs.append(j)
            avg_cost=(avg_cost+j)/(epoch*m +iteration+1)
            if (epochs*m +iteration)%iter_to_print==0:
                print("{} iteration  total average cost {}".format(epoch*m +iteration,avg_cost))
    return parameters,costs

parameters,costs=treenet(trees.copy(),labels,epochs=1,embedding_size=50,beta1=0.9,beta2=0.99,l2_norm=1e-2,learning_rate=1e-2,epsilon=1e-4,iter_to_print=1)
test_tree=np.repeat(trees[1],10)
parameters,costs=treenet(test_tree.copy(),labels,epochs=1,embedding_size=50,beta1=0.9,beta2=0.99,l2_norm=0,learning_rate=1e-2,epsilon=1e-4,iter_to_print=1)


import matplotlib.pyplot as plt
plt.plot(costs)

