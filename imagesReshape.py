# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:20:05 2018

@author: Abhishek S
"""

from PIL import Image
import numpy as np
import pandas as pd
import os
fileloc='E:/Clustering and Retreival/Week 4/images/'

def load_images(directory,shp):
    cd=os.getcwd()
    os.chdir(directory)
    a=np.zeros((shp*shp*3,1))
    k=0
    for i in os.listdir('.'):
        if(k>5):
            break
        k+=1
        print i
        img=Image.open(i)
        img=img.resize((shp,shp),Image.ANTIALIAS)
        imgarray=np.array(img)
        imgarray=imgarray.reshape(-1,1)
        a=np.concatenate((a,imgarray),axis=1)
    os.chdir(cd)
    return a[:,1:]

        
        

cloudy_sky=load_images(fileloc+'cloudy_sky',100)
rivers=load_images(fileloc+'rivers',100)
sunsets=load_images(fileloc+'sunsets',100)
trees_and_forest=load_images(fileloc+'trees_and_forest',100)
X=np.concatenate([cloudy_sky,rivers,sunsets,trees_and_forest],axis=1)
cs=np.repeat(1.,cloudy_sky.shape[1])
ri=np.repeat(1.,rivers.shape[1])
ss=np.repeat(1.,sunsets.shape[1])
tf=np.repeat(1.,trees_and_forest.shape[1])
Y=np.concatenate([cs,np.zeros(ri.shape[0]+ss.shape[0]+tf.shape[0])],axis=0).reshape(1,-1)
np.random.seed(1002)
perm=np.arange(0,Y.shape[1])
np.random.shuffle(perm)
X_train=X[:,perm]
Y_train=Y[:,perm]
X.shape
Y.shape
data=np.concatenate([Y,X],axis=0)

np.save('images',data)



os.chdir('S:/Anaconda/Python_Scripts')
import NN as nn
data=np.load('images.npy')
Y=data[0,:]
X=data[1:,:]
X.shape#(30001L, 1328L)
Y=Y.reshape(1,-1)
Y.shape#(1L, 1328L)
np.random.seed(2001)
from sklearn.model_selection import train_test_split
X_tr,X_te,Y_tr,Y_te=train_test_split(X.T,Y[0],test_size=0.3)
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit_transform(X_tr)
X_train=scalar.fit_transform(X_tr)
X_test=scalar.fit_transform(X_te)

X_train=X_train.T
Y_tr=Y_tr.reshape(1,-1)


import time
start=time.time()
Output,cache,costs=nn.neural_network(X_train,Y_tr,[10,10],['relu','relu','sigmoid'],1e-3,max_iter=1001,seed=10,l2_norm=1e-2,verbose=True,iter_to_print=100)
end=time.time()
import matplotlib.pyplot as plt
import math

start_b=time.time()
Output_b,cache_b,costs_b=neural_network_with_batch(X_train,Y_tr,hidden_layer_list=[4,4,4],activation_list=['relu','relu','tanh','sigmoid'],learning_rate=1e-3,\
                                                   max_iter=100000,batch_size=64,seed=10,l2_norm=5e-2,beta1=0,beta2=0,verbose=True,iter_to_print=100)
end_b=time.time()
print("Time to complete = %.5f mins")%((end_b-start_b)/60.)
print("Training accuracy = %.5f"%(np.sum((Output_b>=0.5)==Y_tr)/(float(len(Y_tr[0])))))
np.argwhere((test_result>=0.5) & (test_result<=0.6))
a=((X_te[328,:]).T).reshape(100,100,3)
plt.imshow(a)

Output_b,cache_b,costs_b=neural_network_with_batch_(X_train,Y_tr,hidden_layer_list=[4,4,4],activation_list=['relu','relu','tanh','sigmoid'],learning_rate=1e-3,\
                                                   max_iter=10000,batch_size=64,seed=10,l2_norm=5e-2,beta1=0,beta2=0,verbose=True,iter_to_print=100)

Output_rms,cache_rms,costs_rms=neural_network_with_batch(X_train,Y_tr,hidden_layer_list=[4,4,4],activation_list=['relu','relu','tanh','sigmoid'],learning_rate=1e-3,\
                                                   max_iter=10000,batch_size=64,seed=10,l2_norm=5e-2,beta1=.9,beta2=.999,verbose=True,iter_to_print=100,epsilon=1e-8)

parameter,_,_=cache_rms
import numpy as np

np.sum(np.isnan(parameter['b3']))
np.sum(np.isnan(parameter['b2']))
np.sum(np.isnan(parameter['b1']))
        AL,linear_cache=nn.forward_propagation_(X_train,[4,4,4],parameter,['relu','relu','tanh','sigmoid'])


plt.plot(costs_rms)

Output_r,cache_r,costs_r=neural_network_with_batch(X_train,Y_tr,hidden_layer_list=[4,4,4],activation_list=['relu','relu','tanh','sigmoid'],learning_rate=1e-3,\
                                                   max_iter=10000,batch_size=64,seed=10,l2_norm=5e-2,beta1=.9,beta2=.999,verbose=True,iter_to_print=100,epsilon=1e-8)

Output_r2,cache_r2,costs_r2=neural_network_with_batch(X_train,Y_tr,hidden_layer_list=[2,2],activation_list=['tanh','tanh','sigmoid'],learning_rate=1e-2,\
                                                   max_iter=10000,batch_size=None,seed=10,l2_norm=5e-2,beta1=.9,beta2=.999,verbose=True,iter_to_print=1,epsilon=1e-8)


Output_r2,cache_r2,costs_r2=neural_network_with_batch(train_X,train_Y,hidden_layer_list=[2,2],activation_list=['tanh','tanh','sigmoid'],learning_rate=1e-2,\
                                                   max_iter=10000,batch_size=None,seed=10,l2_norm=0,beta1=0,beta2=0,verbose=True,iter_to_print=1,epsilon=1e-8)




plt.plot(costs_r)
X_test=X_test.T
Y_te=Y_te.reshape(1,-1)

test_result,_=nn.nn_predict(cache_b,X_test)
np.sum((test_result>=0.5)==Y_te)/float(len(Y_te[0]))
imgarray=imgarray.reshape([100,100,3])
plt.imshow(imgarray)
X_test_orig=scalar.inverse_transform(X_test.T).T
X_test_orig=X_test_orig[:,0].reshape([100,100,3])
plt.imshow(X_test_orig)
X_test_orig.shape
test_result[0,0:10]
original=scalar.inverse_transform(X_test.T).T
original.shape
original_1=original[:,0].reshape(100,100,3)
plt.imshow(Image.fromarray(original_1))
original_1.dtype
np.max(original_1)
imgarray
a=X[:,0].reshape(100,100,3)
plt.imshow(a)
np.max(a)
(end-start)/60
np.power(a,2)

plt.imshow(cloudy_sky[:,0].reshape(100,100,3).astype('uint8'))
a=Image.open(fileloc+'cloudy_sky/ANd9GcQ-yIJezGGCylI2cGGcg9bwTOiVh9iz0mmqI6KVfsdZZqvzBSfQ.jpg')
arr=np.array(a)
plt.imshow(a)
b=a.resize((100,100),Image.ANTIALIAS)
b=(np.array(b))
plt.imshow(b)
c=cloudy_sky[:,1].reshape(100,100,3)
plt.imshow(c)
np.sum(b!=c)
plt.imshow(Image.fromarray(c.astype('uint8')))
mm=np.argwhere((test_result>=0.5)!=Y_te)
plt.imshow((X_te.T)[:,14].reshape(100,100,3).astype('uint8'))
Y_te[0,14]
test_result[0,14]



from sklearn.datasets import make_moons
np.random.seed(3)
train_X, train_Y = make_moons(n_samples=10000, noise=.6) #300 #0.2 

from sklearn.model_selection import train_test_split
X_tr,X_te,Y_tr,Y_te =train_test_split(train_X,train_Y,test_size=0.3)
# Visualize the data
plt.scatter(X_tr[:, 0], X_tr[:, 1], c=Y_tr, s=40, cmap=plt.cm.Spectral);
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_tr)
scaler.mean_
x_tr=scaler.transform(X_tr)
x_te=scaler.transform(X_te)
np.mean(x_tr,axis=0)
np.mean(x_te,axis=0)
x_tr=x_tr.T
y_tr=Y_tr.reshape(1,-1)
x_te=x_te.T
y_te=Y_te.reshape(1,-1)
def accuracy(cache,data,Y):
    parameter,hidden_layer_list,activtion_list=cache
    AL,_=nn.forward_propagation_(data,hidden_layer_list,parameter,activation_list)
    acc=np.sum((AL>=0.5)==Y)/float(Y.shape[1])
    return acc

optimizer=('gd','gdm','adam')
out=dict()
cachel={}
costl={}
hidden_layer_list=[6,6,6]
activation_list=['relu','relu','tanh','sigmoid']
learning_rate=1e-3
l2_norm=1e-1
max_iter=1000
batch_size=64
for op in optimizer:
    out[op],cachel[op],costl[op]=neural_network_with_batch(x_tr,y_tr,hidden_layer_list,activation_list,learning_rate,l2_norm,max_iter,batch_size,\
                              beta1=0.9,beta2=0.999,seed=10,iter_to_print=100,optimization=op,epsilon=1e-8,verbose=True)
    print('----------------------------------------------------------------------------------')
    print("Training accuracy=%0.5f")%(accuracy(cachel[op],x_tr,y_tr))
    print("Test accuracy=%0.5f")%(accuracy(cachel[op],x_te,y_te))
    print('----------------------------------------------------------------------------------')
    
nn.decision_boundary(x_tr,y_tr,cachel['gd'])
nn.decision_boundary(x_tr,y_tr,cachel['gdm'])
nn.decision_boundary(x_tr,y_tr,cachel['adam'])
for op in optimizer:
    plt.plot(costl[op],label=op,alpha=0.7)
plt.legend(loc='best')
plt.xlabel('iteration')
plt.ylabel('Cost per iteration')
plt.show()


























