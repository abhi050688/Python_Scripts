# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 12:03:36 2018

@author: Abhishek S
"""

import pandas as pd
import numpy as np
from scipy.sparse  import csr_matrix
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

wiki=pd.read_csv('E:/Clustering and Retreival/Week 6/people_wiki.csv')
wiki.head()
filename='E:/Clustering and Retreival/Week 6/people_wiki_tf_idf.npz'
def csr_load(filename):
    loader=np.load(filename)
    data=loader['data']
    indptr=loader['indptr']
    indices=loader['indices']
    shape=loader['shape']
    return csr_matrix((data,indices,indptr),shape)

tf_idf=csr_load(filename)
print tf_idf[0,:]
tf_idf.shape

fl=open('E:/Clustering and Retreival/Week 6/people_wiki_map_index_to_word.json')
map_index_to_word=json.load(fl)
fl.close()

tf_idf=normalize(tf_idf)
type(wiki.name)


def bipartition(cluster,maxiter=400,num_runs=4,seed=None):
    dataframe=cluster['dataframe']
    data_matrix=cluster['data_matrix']
    km=KMeans(n_clusters=2,n_init=num_runs,max_iter=maxiter,random_state=seed)
    km.fit(data_matrix)
    centroids,cluster_ass=km.cluster_centers_,km.labels_
    data_matrix_left_child,data_matrix_right_child=data_matrix[cluster_ass==0,:],data_matrix[cluster_ass==1,:]
    cluster_ass=pd.Series(cluster_ass)
    data_frame_left,data_frame_right=dataframe.loc[cluster_ass==0,:],dataframe.loc[cluster_ass==1,:]
    cluster_left={'dataframe':data_frame_left,
                  'data_matrix':data_matrix_left_child,
                  'centroids':centroids[0]}
    cluster_right={'dataframe':data_frame_right,
                   'data_matrix':data_matrix_right_child,
                   'centroids':centroids[1]}
    return cluster_left,cluster_right



wiki_data={'dataframe':wiki,'data_matrix':tf_idf}
left_child,right_child=bipartition(wiki_data,maxiter=100,num_runs=6,seed=1)
mword=pd.DataFrame({'category':map_index_to_word.keys(),'values':map_index_to_word.values()})

map_index_to_word.keys()
map_index_to_word.values()




def display_single_tf_idf_cluster(cluster,mword):
    wiki_subset=cluster['dataframe']
    tf_idf_subset=cluster['data_matrix']
    centroid=cluster['centroids']
    idx=centroid.argsort()[::-1]
    for i in xrange(5):
        print mword[mword.values==idx[i]]
    print ''
    distance=pairwise_distances(tf_idf_subset,[centroid],metric='euclidean').flatten()
    nearest_neighbors=distance.argsort()
    for i in xrange(8):
        text = ' '.join(wiki_subset.iloc[nearest_neighbors[i],:]['text'].split(None, 25)[0:25])
        print('* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki_subset.iloc[nearest_neighbors[i],:]['name'],
              distance[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
    print('')
display_single_tf_idf_cluster(left_child,mword)
cluster=left_child
i=0
athletes = left_child
non_athletes=right_child
athletes['dataframe']=left_child['dataframe'].reset_index(drop=True)
non_athletes['dataframe'] = right_child['dataframe'].reset_index(drop=True)
# Bipartition the cluster of athletes
left_child_athletes, right_child_athletes = bipartition(athletes, maxiter=100, num_runs=6, seed=1)


display_single_tf_idf_cluster(left_child_athletes,mword)
display_single_tf_idf_cluster(right_child_athletes,mword)

ice_hockey_football=right_child_athletes
ice_hockey_football['dataframe']=right_child_athletes['dataframe'].reset_index(drop=True)

left_child_ihs, right_child_ihs = bipartition(ice_hockey_football, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_ihs, mword)
display_single_tf_idf_cluster(right_child_ihs, mword)


# Bipartition the cluster of non-athletes
left_child_non_athletes, right_child_non_athletes = bipartition(non_athletes, maxiter=100, num_runs=6, seed=1)

display_single_tf_idf_cluster(left_child_non_athletes, mword)
display_single_tf_idf_cluster(right_child_non_athletes, mword)

lcna_m=left_child_non_athletes
lcna_m['dataframe']=left_child_non_athletes['dataframe'].reset_index(drop=True)
left_child_non_athletes_m, right_child_non_athletes_m = bipartition(lcna_m, maxiter=100, num_runs=6, seed=1)

display_single_tf_idf_cluster(left_child_non_athletes_m, mword)
display_single_tf_idf_cluster(right_child_non_athletes_m, mword)

rcna_f=right_child_non_athletes
rcna_f['dataframe']=right_child_non_athletes['dataframe'].reset_index(drop=True)
left_child_non_athletes_f, right_child_non_athletes_f = bipartition(rcna_f, maxiter=100, num_runs=6, seed=1)

display_single_tf_idf_cluster(left_child_non_athletes_f, mword)
display_single_tf_idf_cluster(right_child_non_athletes_f, mword)


