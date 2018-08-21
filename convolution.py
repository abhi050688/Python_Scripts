fa
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math
import matplotlib.pyplot as plt


a=np.arange(0,150).reshape(2,3,5,5)
a[1:4,1:4]=0
def conv_forward(X,fltr,padding,stride=1):
    x=X.shape[0]
    f=fltr.shape[0]
    new_x=np.zeros([x+2*padding,x+2*padding])
    new_x[padding:padding+x,padding:padding+x]=X
    o=math.floor((new_x.shape[0]-f)/stride)+1
    output=np.zeros([o,o])
    for i in range(o):
        for j in range(o):
            output[i,j]=np.sum(np.multiply(new_x[i:i+f,j:j+f],fltr))
    return output,new_x
a=np.zeros([6,6])
a[:,0:3]=10
fltr=np.zeros([3,3])
fltr[:,0]=1
fltr[:,1]=0
fltr[:,2]=-1
a[:,3:]=10

fltr=np.array([0,1,-1,0,1,3,-3,-1,1,3,-3,-1,0,1,-1,0]).reshape(4,4)

?np.convolve
a[0]
b,new_x=conv_forward(a,fltr,padding=0,stride=1)
b=np.pad(a,((0,0),(0,0),(1,2),(2,1)),mode='constant',constant_values=0)
a=np.arange(0,np.power(5,4)).reshape(5,5,5,5)
b=np.pad(a, ((0,0),(0,0), (1,1),(3,3)), 'constant', constant_values = 0)
b[0].shape
np.random.seed(1)
a=np.random.randn(4,3,3,2)
b=np.pad(a,((0,0),(1,1),(1,1),(0,0)),'constant',constant_values=0)
b.shape
a[0,:,:,0]
a[0]

b=np.zeros([3,3,3],dtype='uint8')
b[0,:,:]=255
b[:,:,1]=100
b[:,:,2]=100
b=b.astype('unit8')
plt.imshow(b[0,:,:])

a=np.arange(0,75).reshape(5,5,3)
a.reshape(5,-1).T



