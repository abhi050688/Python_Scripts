# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 11:00:25 2018

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances,cosine_distances
from sklearn.metrics.pairwise import pairwise_distances
import json
from itertools import combinations
from copy import copy
import time

def csr_loader(filename):
    loader=np.load(filename)
    data=loader['data']
    indptr=loader['indptr']
    indices=loader['indices']
    shape=loader['shape']
    return csr_matrix((data,indices,indptr),shape)

people=pd.read_csv('E:\Clustering and Retreival\Week 2\people_wiki.csv')
wiki=people.copy()
wiki.head()
tfidf='E:\Clustering and Retreival\Week 2\people_wiki_tf_idf.npz'
corpus=csr_loader(tfidf)
itow='E:\Clustering and Retreival\Week 2\people_wiki_map_index_to_word.json'
fl=open(itow)
map_index_to_word=json.load(fl)
fl.close()

?np.random.randn
np.random.randn(3,2)

def generate_random_vec(num_vec,dim):
    return(np.random.randn(dim,num_vec))

np.random.seed(0)
print generate_random_vec(num_vec=3,dim=5)
np.random.seed(0)
randvec=generate_random_vec(16,547979)
print randvec.shape
doc=corpus[0,:]
print doc.dot(randvec[:,0])>=0
print np.array(doc.dot(randvec)>=0,dtype=int)
print np.array(corpus.dot(randvec)>=0,dtype=int).shape
?np.arange
index_bits=np.array(corpus.dot(randvec)>=0,dtype=int)
power_of_two=1<<np.arange(15,-1,-1)
print index_bits.dot(power_of_two)
hashes=pd.DataFrame({'indices':index_bits.dot(power_of_two)})
type(hashes)
hashes[hashes.indices==143].groupby('indices')['indices'].count()

def train_lsh(data,num_vector,seed=None):
    sh=data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    randvec=generate_random_vec(num_vector,sh)
    power_of_two=1<<np.arange(num_vector-1,-1,-1)
    index_bits=np.array(data.dot(randvec)>=0,dtype=int)
    int_index=index_bits.dot(power_of_two)
    table=dict()
    for data_index,bin_index in enumerate(int_index):
        if bin_index in table:
            table[bin_index].append(data_index)
        else:
            table[bin_index]=[data_index]
    model={'data':data,
           'index_bits':index_bits,
           'int_index':int_index,
           'table':table,
           'num_vector':num_vector,
           'random_vector':randvec
            }
    return model
data=corpus
num_vector=16
seed=143

table[10689]
model=train_lsh(corpus,16,143)
table = model['table']
if   0 in table and table[0]   == [39583] and \
   143 in table and table[143] == [19693, 28277, 29776, 30399]:
    print 'Passed!'
else:
    print 'Check your code.'

wiki[wiki.name.isin(['Barack Obama','Joe Biden'])]
model['index_bits'][[35817,24478]]
print wiki.iloc[model['table'][int_index[35817]],:]
model['int_index'][35817]
search_radius=3

for diff in combinations(range(num_vector),0):
    print diff
q=model['index_bits'][0]
q[15]
query_bin_bits=q
def search_nearby_bins(query_bin_bits,table,\
                       search_radius=2,initial_candidates=set()):
    power_of_two=1<<np.arange(len(query_bin_bits)-1,-1,-1)
#    int_indx=query_bin_bits.dot(power_of_two)
#    initial_candidates=set(list(initial_candidates)+table[int_indx])
    alternate=copy(query_bin_bits)
    candidate_set=copy(initial_candidates)
    for diff in combinations(range(len(query_bin_bits)),search_radius):
#       print diff
        alternate=copy(query_bin_bits)
        for i in diff:
            alternate[i]=abs(alternate[i]-1)
#        print query_bin_bits
#        print alternate
#        print ''
        indx_bit=alternate.dot(power_of_two)
        if indx_bit in table:
            candidate_set.update(table[indx_bit])
    return candidate_set

obama_bin_index = model['index_bits'][35817] # bin index of Barack Obama
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=0)
if candidate_set == set([35817, 21426, 53937, 39426, 50261]):
    print 'Passed test'
else:
    print 'Check your code'
print 'List of documents in the same bin as Obama: 35817, 21426, 53937, 39426, 50261'
            
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=1, initial_candidates=candidate_set)
if candidate_set == set([39426, 38155, 38412, 28444, 9757, 41631, 39207, 59050, 47773, 53937, 21426, 34547,
                         23229, 55615, 39877, 27404, 33996, 21715, 50261, 21975, 33243, 58723, 35817, 45676,
                         19699, 2804, 20347]):
    print 'Passed test'
else:
    print 'Check your code'
print(corpus[[1,2],:])
def query(vec,model,k,max_search_radius):
    table=model['table']
    data=model['data']
    randvec=model['random_vector']
    q=np.array(vec.dot(randvec)>=0,dtype=int)
    q=q.flatten()
    initial_candidates=set()
    candidate_set=copy(initial_candidates)
    for i in xrange(max_search_radius+1):
        candidate_set=search_nearby_bins(q,table,\
                                         search_radius=i,initial_candidates=candidate_set)
    candidates=list(candidate_set)
    docs=corpus[candidates,:]
    nearest_neighbors=pd.DataFrame({'id':candidates})
    nearest_neighbors['distance']=pairwise_distances(vec,docs,metric='cosine').flatten()
    nearest_neighbors.sort_values('distance',ascending=True,inplace=True)
    return nearest_neighbors.iloc[0:10,:],len(candidate_set)

print query(corpus[35817,:],model,k=10,max_search_radius=3)
res,lngth=query(corpus[35817,:],model,k=10,max_search_radius=3)
type(res)
n_res=wiki.join(res.set_index('id'),how='inner')
n_res.sort_values('distance')

num_candidates_history=[]
max_distance_from_query_history=[]
min_distance_from_query_history=[]
average_distance_from_query_history=[]
query_time_history=[]
for max_search_radius in xrange(17):
    start=time.time()
    results,lngth=query(corpus[35817,:],model,k=10,\
                        max_search_radius=max_search_radius)
    print 'Radius:',max_search_radius
    print results.join(wiki[['name']],on='id',how='inner').sort_values('distance')
    end=time.time()
    query_time=end-start
    average_distance_from_query=results['distance'][1:].mean()
    max_distance_from_query=results['distance'][1:].max()
    min_distance_from_query=results['distance'][1:].min()
    num_candidates_history.append(lngth)
    max_distance_from_query_history.append(max_distance_from_query)
    min_distance_from_query_history.append(min_distance_from_query)
    average_distance_from_query_history.append(average_distance_from_query)
    query_time_history.append(query_time)


plt.figure(figsize=(7,4.5))
plt.plot(num_candidates_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('# of documents searched')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(query_time_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance of neighbors')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
for i,j in enumerate(average_distance_from_query_history):
    print i,j

def brute_force_query(vec,data,k):
    shape=data.shape[0]
    nearest_n=pd.DataFrame({'id':range(shape)})
    nearest_n['distance']=pairwise_distances(vec,data,metric='cosine').flatten()
    nearest_n.sort_values('distance',inplace=True)
    return nearest_n.iloc[:25,:]

max_radius=17
precision={i:[] for i in xrange(max_radius)}
average_distance={i:[] for i in xrange(max_radius)}
query_time={i:[] for i in xrange(max_radius)}
np.random.seed(0)
num_queries=10
for i,ix in enumerate(np.random.choice(corpus.shape[0],num_queries,replace=False)):
    print('%s / %s'%(i,num_queries))
    ground_truth=set(brute_force_query(corpus[ix,:],corpus,k=25)['id'])
    for r in xrange(1,max_radius):
        start=time.time()
        result,num_candidates=query(corpus[ix,:],model,k=10,max_search_radius=r)
        end=time.time()
        query_time[r].append(end-start)
        precision[r].append(len(set(result['id']) & ground_truth)/10.0)
        average_distance[r].append(result['distance'][1:].mean())

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(average_distance[i]) for i in xrange(1,17)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(precision[i]) for i in xrange(1,17)], linewidth=4, label='Precison@10')
plt.xlabel('Search radius')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(query_time[i]) for i in xrange(1,17)], linewidth=4, label='Query time')
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()


precision = {i:[] for i in xrange(5,20)}
average_distance  = {i:[] for i in xrange(5,20)}
query_time = {i:[] for i in xrange(5,20)}
num_candidates_history = {i:[] for i in xrange(5,20)}
ground_truth = {}

np.random.seed(0)
num_queries = 10
docs = np.random.choice(corpus.shape[0], num_queries, replace=False)

for i, ix in enumerate(docs):
    ground_truth[ix] = set(brute_force_query(corpus[ix,:], corpus, k=25)['id'])
    # Get the set of 25 true nearest neighbors

for num_vector in xrange(5,20):
    print('num_vector = %s' % (num_vector))
    model = train_lsh(corpus, num_vector, seed=143)
    
    for i, ix in enumerate(docs):
        start = time.time()
        result, num_candidates = query(corpus[ix,:], model, k=10, max_search_radius=3)
        end = time.time()
        
        query_time[num_vector].append(end-start)
        precision[num_vector].append(len(set(result['id']) & ground_truth[ix])/10.0)
        average_distance[num_vector].append(result['distance'][1:].mean())
        num_candidates_history[num_vector].append(num_candidates)

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(average_distance[i]) for i in xrange(5,20)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('# of random vectors')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(precision[i]) for i in xrange(5,20)], linewidth=4, label='Precison@10')
plt.xlabel('# of random vectors')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(query_time[i]) for i in xrange(5,20)], linewidth=4, label='Query time (seconds)')
plt.xlabel('# of random vectors')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(num_candidates_history[i]) for i in xrange(5,20)], linewidth=4,
         label='# of documents searched')
plt.xlabel('# of random vectors')
plt.ylabel('# of documents searched')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()










