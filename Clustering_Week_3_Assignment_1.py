# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:00:47 2018

@author: Abhishek S
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import sys
import os
import json
import matplotlib.pyplot as plt

wiki=pd.read_csv('E:/Clustering and Retreival/Week 3/people_wiki.csv')
wiki.head()

def load_sparse(filename):
    loader=np.load(filename)
    indptr=loader['indptr']
    data=loader['data']
    indices=loader['indices']
    shape=loader['shape']
    return csr_matrix((data,indices,indptr),shape)
tf_idf=load_sparse('E:/Clustering and Retreival/Week 3/people_wiki_tf_idf.npz')
fl=open('E:/Clustering and Retreival/Week 3/people_wiki_map_index_to_word.json')
map_index_to_word=json.load(fl)
fl.close()
?normalize
tf_idf=normalize(tf_idf)

np.random.randint(0,10,4)
def get_initial_centroids(data,k,seed=None):
    if seed is not None:
        np.random.seed(seed)
    mx=data.shape[0]
    centre=np.random.randint(0,mx,k)
    centroids=data[centre,:].toarray()
    return centroids

queries=tf_idf[100:102,:]
dist=pairwise_distances(tf_idf,queries,metric='euclidean')
print dist.shape
dist=pairwise_distances(tf_idf,tf_idf[:3,:],metric='euclidean')
'''Test cell'''
if np.allclose(dist[430,1], pairwise_distances(tf_idf[430,:], tf_idf[1,:])):
    print('Pass')
else:
    print('Check your code again')

closest_cluster=np.argmin(dist,axis=1)
'''Test cell'''
reference = [list(row).index(min(row)) for row in dist]
if np.allclose(closest_cluster, reference):
    print('Pass')
else:
    print('Check your code again')
?np.bincount

np.bincount(closest_cluster)
def assign_clusters(data,centroids):
    distance=pairwise_distances(data,centroids,metric='euclidean')
    closest_cluster=np.argmin(distance,axis=1)
    return closest_cluster

if np.allclose(assign_clusters(tf_idf[0:100:10], tf_idf[0:8:2]), np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1])):
    print('Pass')
else:
    print('Check your code again.')
dt=dist[:2,:]
np.mean(dt,axis=1)
np.mean(dt,axis=0)

def revise_centroids(data,k,cluster_assignment):
    new_centroids=[]
    for i in xrange(k):
        new_centroids.append(np.array(np.mean(data[cluster_assignment==i,:],axis=0)).flatten())
    return np.array(new_centroids)


a=np.mean(data[cluster_assignment==0,:],axis=0)
a.shape
np.array(new_centroids).shape

result=result[:,0,:]

?np.array
result.shape
result.flatten()
result = revise_centroids(tf_idf[0:100:10], 3, np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1]))
if np.allclose(result[0], np.mean(tf_idf[[0,30,40,60]].toarray(), axis=0)) and \
   np.allclose(result[1], np.mean(tf_idf[[10,20,90]].toarray(), axis=0))   and \
   np.allclose(result[2], np.mean(tf_idf[[50,70,80]].toarray(), axis=0)):
    print('Pass')
else:
    print('Check your code')
cluster=new_cluster
def compute_heterogeneity(data,k,cluster,centroids):
    td=0.0
    for i in xrange(k):
        dt=data[cluster==i,:]
        if dt.shape[0]!=0:
            dist=pairwise_distances(dt,centroids[i,:].reshape((1,data.shape[1])),metric='euclidean')
            dist=dist**2
            td+=np.sum(dist)
    return td

def kmeans(data,k,initial_centroids,max_iter,record_heter,verbose=False):
    centroids=initial_centroids[:]
    prev_cluster=None
    for i in xrange(max_iter):
#        print('iteration %d'%(i))
        new_cluster=assign_clusters(data,centroids)
        print new_cluster
        centroids=revise_centroids(data,k,new_cluster)
        if prev_cluster is not None:
            if (prev_cluster==new_cluster).all():
                break
        if prev_cluster is not None:
            cng=sum(prev_cluster!=new_cluster)
            if verbose:
                print('{0:5d} elements changed their cluster assignment'.format(cng))
        if record_heter is not None:
            score=compute_heterogeneity(data,k,new_cluster,centroids)
            record_heter.append(score)
        prev_cluster=new_cluster.copy()
    return centroids,prev_cluster


def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

k=3
heterogeneity_i=[]
initial_centroids=get_initial_centroids(tf_idf,k,seed=0)        
centr,cluster_assignment11=kmeans(tf_idf,k,initial_centroids,max_iter=400,record_heter=heterogeneity_i,verbose=True)
np.bincount(cluster_assignment11)
plot_heterogeneity(heterogeneity_i,k)



k = 10
heterogeneity = {}
import time
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = get_initial_centroids(tf_idf, k, seed)
    centroids, cluster_assignment1 = kmeans(tf_idf, k, initial_centroids, max_iter=400,
                                           record_heter=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity[seed] = compute_heterogeneity(tf_idf, k,cluster_assignment1, centroids)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
    print np.bincount(cluster_assignment1)
    sys.stdout.flush()
end = time.time()
print(end-start)
set(cluster_assignment)



data=tf_idf
data.shape
centroids.shape
new_cluster.shape
nn=revise_centroids(data,k,new_cluster)
nn.shape
cluster_assignment=new_cluster

def smart_initialization(data,k,seed=None):
    if seed is not None:
        np.random.seed(seed)
    centroids=np.zeros((k,data.shape[1]))
    idx=np.random.randint(data.shape[0])
    centroids[0]=data[idx,:].toarray()
    squared_dist=pairwise_distances(data,centroids[0:1],metric='euclidean').flatten()**2
    for i in xrange(1,k):
        idx=np.random.choice(data.shape[0],1,p=squared_dist/sum(squared_dist))
        centroids[i]=data[idx,:].toarray()
        squared_dist=np.min(pairwise_distances(data,centroids[0:1+1],metric='euclidean'),axis=1)
    return centroids

seed=0
k = 10
heterogeneity_smart = {}
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = smart_initialization(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, max_iter=400,
                                           record_heter=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, cluster_assignment, centroids)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity_smart[seed]))
    sys.stdout.flush()
end = time.time()
print(end-start)

plt.figure(figsize=(8,5))
plt.boxplot([heterogeneity.values(), heterogeneity_smart.values()], vert=False)
plt.yticks([1, 2], ['k-means', 'k-means++'])
plt.rcParams.update({'font.size': 16})
plt.tight_layout()


def kmeans_multiple_runs(data,k,maxiter,num_runs,seed_list=None,verbose=False):
    heterogeneity={}
    min_heterogeneity_achieved=float('inf')
    best_seed=None
    final_centroids=None
    final_cluster_assignment=None
    for i in xrange(num_runs):
        if seed_list is not None:
            seed=seed_list[i]
        else:
            seed=int(time.time())
            np.random.seed(seed)
        initial_centroids=smart_initialization(data,k,seed=seed)
        centroids,cluster_assignment=kmeans(data,k,initial_centroids,maxiter,record_heter=None,verbose=False)
        heterogeneity[seed]=compute_heterogeneity(data,k,cluster_assignment,centroids)
        if verbose:
            print('seed = %s , heterogeneity= %s '%(seed,heterogeneity[seed]))
            sys.stdout.flush()
        if heterogeneity[seed]<min_heterogeneity_achieved:
            min_heterogeneity_achieved=heterogeneity[seed]
            best_seed=seed
            final_centroids=centroids
            final_cluster=cluster_assignment
    return final_centroids,final_cluster_assignment




def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    plt.figure(figsize=(7,4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

filename = 'E:/Clustering and Retreival/Week 3/kmeans-arrays.npz'

heterogeneity_values = []
k_list = [2, 10, 25, 50, 100]

if os.path.exists(filename):
    arrays = np.load(filename)
    centroids = {}
    cluster_assignment = {}
    for k in k_list:
        print k
        sys.stdout.flush()
        centroids[k] = arrays['centroids_{0:d}'.format(k)]
        cluster_assignment[k] = arrays['cluster_assignment_{0:d}'.format(k)]
        score = compute_heterogeneity(tf_idf, k, cluster_assignment[k], centroids[k])
        heterogeneity_values.append(score)
    
    plot_k_vs_heterogeneity(k_list, heterogeneity_values)

else:
    print('File not found. Skipping.')

def visualize_document_clusters(wiki, tf_idf, centroids, cluster_assignment, k, map_index_to_word, display_content=True):
    '''wiki: original dataframe
       tf_idf: data matrix, sparse matrix format
       map_index_to_word: SFrame specifying the mapping betweeen words and column indices
       display_content: if True, display 8 nearest neighbors of each centroid'''
    
    print('==========================================================')

    # Visualize each cluster c
    for c in xrange(k):
        # Cluster heading
        print('Cluster {0:d}    '.format(c)),
        # Print top 5 words with largest TF-IDF weights in the cluster
        idx = centroids[c].argsort()[::-1]
#        for i in xrange(5): # Print each word along with the TF-IDF weight
#            print('{0:s}:{1:.3f}'.format(map_index_to_word['category'][idx[i]], centroids[c,idx[i]])),
#        print('')
        
        if display_content:
            # Compute distances from the centroid to all data points in the cluster,
            # and compute nearest neighbors of the centroids within the cluster.
            distances = pairwise_distances(tf_idf, centroids[c].reshape(1, -1), metric='euclidean').flatten()
            distances[cluster_assignment!=c] = float('inf') # remove non-members from consideration
            nearest_neighbors = distances.argsort()
            # For 8 nearest neighbors, print the title as well as first 180 characters of text.
            # Wrap the text at 80-character mark.
            for i in xrange(8):
                text = ' '.join(wiki.iloc[nearest_neighbors[0],2].split(None, 25)[0:25])
                print('\n* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki.iloc[nearest_neighbors[i],1],
                    distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
        print('==========================================================')
visualize_document_clusters(wiki, tf_idf, centroids[2], cluster_assignment[2], 2, map_index_to_word)    



k = 10
visualize_document_clusters(wiki, tf_idf, centroids[k], cluster_assignment[k], k, map_index_to_word)
sum(np.bincount(cluster_assignment[100])<236)


point=np.array([-1.88,2.05,-0.71,0.42,2.41,-0.67,1.85,-3.80,-3.69,-1.33]).reshape((5,2))
centro=np.array([2.0,2.0,-2.0,-2.0]).reshape((2,2))
c,pc=kmeans(point,2,centro,400,None,True)

