# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:07:00 2018

@author: Abhishek S
"""

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


folder="E:/Clustering and Retreival/Week 4/images/"

#os.getcwd()
#os.chdir()
#img=Image.open(folder+'cloudy_sky/'+'ANd9GcQ0sa93MklyNXW_oukodvV0P1-Wyl_bpOLmibpxxbYEkkl4_2Mc.jpg')
#imga=np.array(img)
#a,b,c=imga.shape
#plt.imshow(img)
#np.linspace(0,a,num=8,dtype=int,endpoint=False)
#imga[166:194,:,0]




def get_image_array(fldr):
    cd=os.getcwd()
    try:
        os.chdir(folder+fldr)
        red=list()
        blue=list()
        green=list()
        for i in os.listdir('.'):
            img=Image.open(i)
            imgarray=np.array(img)
            red.append(np.mean(imgarray[:,:,0]))
            blue.append(np.mean(imgarray[:,:,2]))
            green.append(np.mean(imgarray[:,:,1]))
        imndf=pd.DataFrame({'folder':fldr,'red':red,'blue':blue,'green':green})
    except Exception as e:
        os.chdir(cd)
        print e
    return(imndf)

cloudy_sky=get_image_array('cloudy_sky')
cloud.head()
sunsets=get_image_array('sunsets')
trees_and_forest=get_image_array('trees_and_forest')
rivers=get_image_array('rivers')        

imgarray=pd.concat([cloudy_sky,sunsets,trees_and_forest,rivers])
imgarray.head()
imgarray.shape
np.random.seed(101)
imgarray=imgarray.sample(frac=1)
imgarray.reset_index(drop=True,inplace=True)
imgmat=np.array(imgarray[['red','green','blue']])

km_model=KMeans(n_clusters=4,init='k-means++',n_init=4,max_iter=1000)
km_model.fit(imgmat)
centroids,clusters=km_model.cluster_centers_,km_model.labels_
imgarray['km']=clusters
imgarray.groupby(['km','folder'])['km'].count()
data=imgmat[:,:]
col=dict(zip(list(np.arange(0,4)),['red','green','blue','orange']))
imgarray['c']=imgarray['km'].map(col)

plt.figure()
plt.scatter(imgarray.red,imgarray.green,c=imgarray.c_gm)
plt.plot([x[0] for x in centroids_gm],[y[1] for y in centroids_gm],'ko')


gm_model=GaussianMixture(n_components=4,covariance_type='diag',max_iter=1000,means_init=centroids,n_init=4)
gm_model.fit(imgmat)
centroids_gm,cluster_gm=gm_model.means_,gm_model.predict(imgmat)

imgarray['km_gm']=cluster_gm
imgarray.groupby(['km_gm','folder'])['km_gm'].count()
imgarray['c_gm']=imgarray['km_gm'].map(col)

np.linspace(0,256,num=8,dtype=int)
f=os.listdir(folder+'sunsets')
len(f)

np.random.seed(0)
def create_features(fldr,brk):
    cd=os.getcwd()
    try:
        os.chdir(folder+fldr)
        mat=np.zeros((len(os.listdir('.')),brk-1,3))
        n_mat=np.zeros([len(os.listdir('.')),(brk-1)*3])
        for i,nm in enumerate(os.listdir('.')):
            img=Image.open(nm)
            img_array=np.array(img)
            a,b,c=img_array.shape
            rw=np.linspace(0,a,num=brk,dtype=int)
            cl=np.linspace(0,b,num=brk,dtype=int)
            for j in xrange(brk-1):
                mat[i,j,0]=np.mean(img_array[rw[j]:rw[j+1],cl[j]:cl[j+1],0])
                mat[i,j,1]=np.mean(img_array[rw[j]:rw[j+1],cl[j]:cl[j+1],1])
                mat[i,j,2]=np.mean(img_array[rw[j]:rw[j+1],cl[j]:cl[j+1],2])
            n_mat[i,:]=np.concatenate([mat[i,:,0],mat[i,:,1],mat[i,:,2]],axis=0)
        os.chdir(cd)
    except Exception as e:
        os.chdir(cd)
        print e
    return n_mat
brk=8                
sunsets_f=create_features('sunsets',brk)
trees_f=create_features('trees_and_forest',brk)
cloudy_f=create_features('cloudy_sky',brk)
rivers_f=create_features('rivers',brk)

i_clus=np.concatenate([np.repeat(0,sunsets_f.shape[0]),np.repeat(1,trees_f.shape[0]),np.repeat(2,cloudy_f.shape[0]),np.repeat(3,rivers_f.shape[0])],axis=0)
i_clus.shape

mat_f=np.concatenate([sunsets_f,trees_f,cloudy_f,rivers_f],axis=0)
mat_f.shape
mat_df=pd.DataFrame(mat_f)
mat_df['cluster']=i_clus
mat_df.head()
mat_df=mat_df.sample(frac=1)
mat_df.reset_index(drop=True,inplace=True)
mat_f=np.array(mat_df.iloc[:,0:((brk-1)*3)-1])
gm_model_f=GaussianMixture(n_components=4,covariance_type='diag',n_init=10,max_iter=10000)
gm_model_f.fit(mat_f)
centroids_f,cluster_f=gm_model_f.means_,gm_model_f.predict(mat_f)
mat_df['gm_f']=cluster_f
fld_m={0:'sunsets',1:'trees',2:'cloud',3:'rivers'}
mat_df['folder']=mat_df['cluster'].map(fld_m)
mat_df.groupby(['gm_f','folder'])['gm_f'].count()

