# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:18:57 2018

@author: Abhishek S
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import json
from sklearn.neighbors import NearestNeighbors
import operator
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt



wiki=pd.read_csv('E:\Clustering and Retreival\Week 2\people_wiki.csv')
wiki.head()
wiki.iloc[1,2]

#?csr_matrix
def csr_loader(filename):
    loader=np.load(filename)
    data=loader['data']
    indices=loader['indices']
    indptr=loader['indptr']
    shape=loader['shape']
    return(csr_matrix((data,indices,indptr),shape))
filename='E:\Clustering and Retreival\Week 2\people_wiki_word_count.npz'
mtrx=csr_loader(filename)
print(mtrx)
itow='E:\Clustering and Retreival\Week 2\people_wiki_map_index_to_word.json'
fl=open(itow)
map_index_to_word=json.load(fl)
fl.close()
print(map_index_to_word)
len(map_index_to_word.keys())
mtrx[35817]
print(wiki[wiki['name']=='Barack Obama'])

model=NearestNeighbors(metric='euclidean',algorithm='brute')
model.fit(mtrx)
distancea,indices=model.kneighbors(mtrx[35817],n_neighbors=10)
neighbors=pd.DataFrame({'distance':distancea.flatten(),'indices':indices.flatten()})

?pd.DataFrame.merge
wiki.merge(neighbors,how='inner',left_index=True,right_on='indices').sort_values('distance')
?pd.DataFrame.join
#Sorting dictionaries based on values
#!By values, sorted(map_index_to_words.items(),key=operator.itemgetter(1)) or sorted(map_index_to_words,key=map_index_to_words.get)
#!By Keys, sorted(map_index_to_words,key=operator.itemgetter(0))

table=sorted(map_index_to_word,key=map_index_to_word.get)
#Table has words sorted on frequency

for i in xrange(10):
    print(map_index_to_word.get(table[i]),table[i])




def unpack_dict(matrix,map_dict_words):
    tabel=sorted(map_dict_words,key=map_dict_words.get)
    indices=matrix.indices
    data=matrix.data
    indptr=matrix.indptr
    docs=matrix.shape[0]
    return([{k:v for k,v in zip( [table[word_id] for word_id in indices[indptr[i]:indptr[i+1]].tolist()],\
                                data[indptr[i]:indptr[i+1]]) } for i in xrange(docs)])

word_count=unpack_dict(mtrx,map_index_to_word)
wiki['word_count']=word_count

def top_words(name):
    row=wiki[wiki['name']==name]
    word_count_vec=row['word_count'].apply(pd.Series).stack()
    word_count_vec=word_count_vec.reset_index()
    del word_count_vec['level_0']
    word_count_vec=word_count_vec.rename(columns={'level_1':'word',0:'count'})
    return(word_count_vec.sort_values('count',ascending=False))

name='Barack Obama'
obama_words=top_words('Barack Obama')
barrio_words=top_words('Francisco Barrio')
print barrio_words
combined=obama_words.merge(barrio_words,on='word')
combined=combined.rename(columns={'count_x':'Obama_count','count_y':'barrio_count'})
top5=combined.word[0:5]
wiki['has_top_words']=wiki['word_count'].apply(lambda x:set(top5).issubset(set(x.keys())))
previous_wiki.iloc[[32,33],:].head()
wiki.has_top_words.sum()
wiki[wiki.name.isin(['Barack Obama','Joe Biden'])]

b_g=euclidean_distances(mtrx[35817],mtrx[28447])
b_j=euclidean_distances(mtrx[35817],mtrx[24478])
j_g=euclidean_distances(mtrx[24478],mtrx[28447])

george_words=top_words('George W. Bush')
c_b_g=obama_words.join(george_words.set_index('word'),how='inner',on='word',lsuffix='Barack',rsuffix='George')
tfidf='E:\Clustering and Retreival\Week 2\people_wiki_tf_idf.npz'
tf_idf=csr_loader(tfidf)
model_tfidf=NearestNeighbors(metric='euclidean',algorithm='brute')
model_tfidf.fit(tf_idf)
distance_tf,index_tf=model_tfidf.kneighbors(tf_idf[35817],n_neighbors=10)
neighbors_tf=pd.DataFrame({'distance':distance_tf.flatten(),'indices':index_tf.flatten()})
wiki.merge(neighbors_tf,left_index=True,right_on='indices',how='inner').sort_values('distance')
wiki['tf_idf']=unpack_dict(tf_idf,map_index_to_word)
previous_wiki=wiki.copy()
del wiki['word_count']
del wiki['has_top_words']
wiki.head()
wiki=wiki.rename(columns={'tf_idf':'word_count'})
obama_tf_idf = top_words('Barack Obama')
schiliro_tf_idf = top_words('Phil Schiliro')
combined_tf=obama_tf_idf.join(schiliro_tf_idf.set_index('word'),on='word',lsuffix='Obama',rsuffix='Schiliro',how='inner')
top5tf=combined_tf['word'][0:5]
wiki['has_top_tf_idf']=wiki['word_count'].apply(lambda x:set(top5tf).issubset(set(x.keys())))
sum(wiki.has_top_tf_idf)
b_j_tf=euclidean_distances(tf_idf[35817],tf_idf[24478])
def compute_len(text):
    return(len(text.split(' ')))
wiki['length']=wiki['text'].apply(compute_len)


distance_100,n_100=model_tfidf.kneighbors(tf_idf[35817],n_neighbors=100)   
n_neighbors_100=pd.DataFrame({'distance':distance_100.flatten(),'indices':n_100.flatten()})
nearest_neighbors_euclidean=wiki.merge(n_neighbors_100,right_on='indices',left_index=True,how='inner')
nearest_neighbors_euclidean=nearest_neighbors_euclidean.rename(columns={'indices':'id'})
nearest_neighbors_euclidean=nearest_neighbors_euclidean[['id','name','length','distance']]






plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.axvline(x=wiki.iloc[35817,5], color='k', linestyle='--', linewidth=4,label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki.iloc[28447,5], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
tf_wiki=wiki.copy()
model2=NearestNeighbors(metric='cosine',algorithm='brute')
model2.fit(tf_idf)
dist_c,n_c=model2.kneighbors(tf_idf[35817],n_neighbors=100)
neigh_c=pd.DataFrame({'distance':dist_c.flatten(),'indices':n_c.flatten()})
nearest_neighbors_cosine=wiki.merge(neigh_c,right_on='indices',left_index=True,how='inner')
nearest_neighbors_cosine=nearest_neighbors_cosine.sort_values('distance')
nearest_neighbors_cosine.head()




a='The quick brown fox jumps over the lazy dog'
b='A quick brown dog outpaces a quick fox'
a=a.lower()
b=b.lower()
a=a.split(' ')
b=b.split(' ')
c=a+b
c=list(set(c))

def generate_random(num_vector,dim):
    return(np.random.randn(dim,num_vector))
print generate_random(num_vector=3,dim=5)




