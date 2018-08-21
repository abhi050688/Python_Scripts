# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:50:29 2018

@author: Abhishek S
"""

import os
from PIL import Image
import numpy as np
import pandas as pd

cloud='E:/Clustering and Retreival/Week 4/images/cloudy_sky'
#for i in os.listdir('.'):
#    print i
#
#
#with Image.open(cloud+'/'+fls[0],'r') as img:
#    print img
#    
img=Image.open(cloud+'/'+cloudyimages.filename[0])
img.show()
img.convert('L')
img.show()
img.shape
img.save('E:/Clustering and Retreival/Week 4/images/0/0.jpg')
#dir(img.getdata)
#img.getdata
#imgarray=np.array(img)
#imgarray.shape
#b=imgarray[:,:,1]
#imgarray[:,:,1]=b+10
#Image.fromarray(imgarray)
#Image.fromarray(np.array(img))
#np.mean(imgarray[:,:,0])


#os.getcwd()
def create_image_array(directory):
    cd=os.getcwd()
    os.chdir(directory)
    r=list()
    b=list()
    g=list()
    fn=list()
    for n in os.listdir('.'):
#        print n
        with Image.open(n) as img:
            imgarray=np.array(img)
            r.append(np.mean(imgarray[:,:,0]))
            b.append(np.mean(imgarray[:,:,2]))
            g.append(np.mean(imgarray[:,:,1]))
            fn.append(n)
    os.chdir(cd)
    return pd.DataFrame({'directory':directory,'filename':fn,'red':r,'green':g,'blue':b})
cloudyimages=create_image_array(cloud)
river=cloud[:cloud.rfind('/')+1]+'rivers'
sunsets=cloud[:cloud.rfind('/')+1]+'sunsets'
trees_and_forest=cloud[:cloud.rfind('/')+1]+'trees_and_forest'
riverimages=create_image_array(river)
sunsetsimages=create_image_array(sunsets)
treesforestimages=create_image_array(trees_and_forest)
treesforestimages.head()
images=pd.concat([cloudyimages,riverimages,sunsetsimages,treesforestimages])
images.shape
images.head(10)
images.reset_index(drop=True,inplace=True)
np.random.seed(190)
images=images.sample(frac=1)
images.reset_index(drop=True,inplace=True)
#images.to_csv('E:/Clustering and Retreival/Week 4/images/imagesarray.csv')
#cen=np.random.choice(images.shape[0],size=4,replace=False)
#initial_centres=images.sample(n=4,replace=False)
fldr=images['directory'].unique()
mp=dict(zip(fldr,[1,2,3,4]))
images['cor_cluster']=images['directory'].map(mp)
images.head()
initial_centres=images.sample(n=4,replace=False)
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(2,2,1)
ax.hist(cloudyimages.blue,30)
plt.title('abc')
plt.show()

def plothist(df,tle):
    fig=plt.figure()
    ax=fig.add_subplot(2,2,1)
    ax.hist(df.blue,30,color='blue')
    ax.set(ylabel=tle)
    ax1=fig.add_subplot(2,2,2)
    ax1.hist(df.red,30,color='red')
    ax1.set(ylabel=tle)
    ax2=fig.add_subplot(2,2,3)
    ax2.hist(df.green,30,color='green')
    ax2.set(ylabel=tle)
    ax.set_xlim(0,250)
    ax1.set_xlim(0,250)
    ax2.set_xlim(0,250)
    plt.show()

plothist(cloudyimages,'Clouds')
plothist(riverimages,'rivers')
plothist(sunsetsimages,'sunsets')
plothist(treesforestimages,'treesforest')

#f,ax=plt.subplots(2,2)
#for i in ax.flat:
#    print i

from sklearn.metrics import pairwise_distances
rgb=['red','green','blue']
#dist=pairwise_distances(images[rgb].head(),initial_centres[rgb],metric='euclidean')
#np.min(dist,axis=1)

def cost_dist(distances):
    return np.sum(np.min(distances,axis=1))
k=4
def newcenters(data,clusters,k):
    arr=np.zeros([k,data.shape[1]])
    for j in set(clusters):
        arr[j,:]=np.mean(data[clusters==j,:],axis=0)
    return arr


def kmeans(df,initial_centres,k,maxiter=1000,verbose=True):
#    print 'asa'
    ini_cen=initial_centres.copy()
    lc=list()
    clusterold=np.zeros(df.shape[0])
    for i in xrange(maxiter):
#        print 'bv'
#        print df
#        print ini_cen
        distance=pairwise_distances(df,ini_cen,metric='euclidean')
        cost=cost_dist(distance)
        lc.append(cost)
        clusternew=np.argmin(distance,axis=1)
        ini_cen=newcenters(df,clusternew,k)
        if verbose:
            if i>0:
                print "Iteration %4d, Cost=%10.3f and cluster changes=%4d"%(i,cost,np.sum(clusterold!=clusternew))
            else:
                print "Iteration %4d, Cost=%10.3f"%(i,cost)
        if np.sum(clusterold!=clusternew)==0:
            break
        clusterold=clusternew
    return ini_cen,clusternew,lc



data=np.array(images[rgb])
ini_cen=np.array(initial_centres[rgb])
centroid,clusters,cst=kmeans(data,ini_cen,4,1000)
images.head()
images['kmeans']=list(clusters)

img.close()

def saveimg(df,k,c):
    cd=os.getcwd()
    df.reset_index(inplace=True)
    for i in xrange(k):
        direct=df.directory[i]
        os.chdir(direct)
        fl=df.filename[i]
        fln=direct[direct.rfind('/')+1:]
        img=Image.open(direct+'/'+fl)
        img.save('E:/Clustering and Retreival/Week 4/images/'+str(c)+'/'+fln+str(i)+'.jpg')
        img.close()
    os.chdir(cd)
saveimg(images.loc[images.kmeans==0,:],20,0)
saveimg(images.loc[images.kmeans==1,:],20,1)
saveimg(images.loc[images.kmeans==2,:],20,2)
saveimg(images.loc[images.kmeans==3,:],20,3)

images.groupby(['kmeans','cor_cluster'])['kmeans'].count()

def smart_initialization(df,k,seed):
    np.random.seed(seed)
    arr=np.zeros([k,df.shape[1]])
    for i in xrange(k):
        if i ==0:
            ch=np.random.choice(df.shape[0],size=1)
            arr[i,:]=df[ch,:]
        else:
            cluster=arr[:i,:]
            distance=pairwise_distances(df,cluster,metric='euclidean')
            mdistance=np.min(distance,axis=1)
            mdistance=np.power(mdistance,2)
            ch=np.random.choice(df.shape[0],size=1,p=mdistance/np.sum(mdistance))
            arr[i,:]=df[ch,:]
    return arr
df=data
k=4
i=1

ini_cent=smart_initialization(df,k,12)
ini_cent=smart_initialization(df,k,100)

centroid_s,cluster_s,cst_s=kmeans(data,ini_cent,4,300)
images['smart']=list(cluster_s)
saveimg(images.loc[images.smart==0,:],20,0)
saveimg(images.loc[images.smart==1,:],20,1)
saveimg(images.loc[images.smart==2,:],20,2)
saveimg(images.loc[images.smart==3,:],20,3)





images.groupby(['smart','cor_cluster'])['smart'].count()
images.groupby(['kmeans','cor_cluster'])['kmeans'].count()












