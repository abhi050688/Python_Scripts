# -*- coding: utf-8 -*-
"""
Created on Sat May 26 11:12:57 2018

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import os
import datetime as dt
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


os.chdir('E:/Analytics Vidya/research')
train=pd.read_csv('information_train.csv',sep='\t')
tr_c=train.columns
col_cat=dict(zip(tr_c,np.repeat(np.object,tr_c.shape[0])))
col_cat['set']=np.int64
del col_cat['pub_date']
parser=lambda date:dt.datetime.strptime(date,"%Y-%m-%d")
train=pd.read_csv('information_train.csv',sep='\t',dtype=col_cat,date_parser=parser,parse_dates=['pub_date'])
train.tail()
train.info()
~train['full_Text'].isnull().all()
test=pd.read_csv('information_test.csv',sep='\t',dtype=col_cat,date_parser=parser,parse_dates=['pub_date'])
train.full_Text.isnull().all()
test.tail()
train.set.unique()
test.set.unique()
train.shape
target_train=pd.read_csv("train.csv",dtype={'pmid':np.object,'ref_list':np.object})
target_train.info()
target_train.head()
full=train.merge(target_train,left_on='pmid',right_on='pmid',how='left')
full.head()
full.info()
exp=full[['pmid','ref_list']]

pmids=list()
def explode(row):
    nrow=ast.literal_eval(row['ref_list'])
    for nn in list(nrow):
        pmids.append([row['pmid'],nn])
_=exp.apply(explode,axis=1)
n_exp=pd.DataFrame(pmids,columns=['pmid','ref_list_split'])
n_exp.head(20)
nfull=full.merge(n_exp,on='pmid')
nfull.head(20)
del nfull['ref_list']
nfull=nfull.merge(full[['pmid','pub_date','author_str']],left_on='ref_list_split',right_on='pmid',suffixes=('','ref_'),how='left')
a=nfull.pub_date>=nfull.pub_dateref_
sum(a)/a.shape[0]
full.info()
a=nfull.groupby('pmid')['pmid'].count()
a.describe()
vect=TfidfVectorizer(stop_words='english',max_df=0.5)
vect=TfidfVectorizer(stop_words='english',max_df=0.5)
tr=vect.fit_transform(full.abstract)
tr.shape
b=cosine_similarity(tr)
pairwise_distances(tr[5,:],tr[21,:],metric='euclidean')
pairwise_distances(tr[2,:],tr[3,:],metric='euclidean').flatten()[0]
nfull_copy=nfull.copy()
full_copy=full.copy()


nfull_copy['euclidean']=nfull_copy.apply(lambda row:pairwise_distances(tr[full_copy[full_copy.pmid==row['pmid']].index.tolist(),:],\
                                                                    tr[full_copy[full_copy.pmid==row['ref_list_split']].index.tolist(),:],metric='euclidean').flatten()[0],axis=1)

nfull_copy['cosine']=nfull_copy.apply(lambda row:pairwise_distances(tr[full_copy[full_copy.pmid==row['pmid']].index.tolist(),:],\
                                                                    tr[full_copy[full_copy.pmid==row['ref_list_split']].index.tolist(),:],metric='cosine').flatten()[0],axis=1)
nfull_copy[['euclidean','cosine']].describe()
nfull_copy[['euclidean','cosine']].quantile(q=0.15)

nbrs=NearestNeighbors(n_neighbors=10,algorithm='brute',metric='euclidean')
nbrs.fit(tr)
distance,indices=nbrs.kneighbors()
distance[3,:]
indices[3,:]

def candidates(row):
    set_ref_=set(full.index[(full.set==row['set']) & (full.pub_date<row['pub_date'])])
    return set_ref_

row=full.iloc[0,:]
candidates(row)

full['candidates']=full[['set','pub_date']].apply(candidates,axis=1)['set']
full.head()
full['nn']=pd.DataFrame(indices).apply(lambda x:set(x),axis=1)
full['nn_c']=full.apply(lambda x:set(x['candidates']).intersection(x['nn']),axis=1)['nn']

a=set(np.arange(0,10))
b=set([3,4,5])
a=list(a)
a[0:15]
full['ref_list'].apply(lambda x:len(ast.literal_eval(x))).describe()

full['nn_c_short']=full['nn_c'].apply(lambda x:set(list(x)[0:8]))
target_train.head()

pmids=list()
_=target_train.apply(explode,axis=1)
npmid=pd.DataFrame(pmids,columns=['pmid','ref_list'])
npmid.head()
fpmid=full[['pmid']]
fpmid.index.name='idx'
fpmid.reset_index(inplace=True)
fpmid.head()
npmid=npmid.merge(fpmid,how='left',left_on='ref_list',right_on='pmid')

nset=npmid.groupby('pmid_x')['idx'].apply(lambda seris:set(seris.values))
nset.name='actuals'
nset=pd.DataFrame(nset)
nset.reset_index(inplace=True)
nset.head()
nfull=full.merge(nset,left_on='pmid',right_on='pmid_x',how='left')
nfull.head()

nfull['precision']=nfull.apply(lambda x:len(x['actuals'].intersection(x['nn_c_short']))/len(x['nn_c_short']) if len(x['nn_c_short'])>0  else 0,axis=1)
nfull['recall']=nfull.apply(lambda x:len(x['actuals'].intersection(x['nn_c_short']))/len(x['actuals']) if len(x['actuals'])>0  else 0,axis=1)
nfull['f1_Score']=2*nfull.precision*nfull.recall/(nfull.precision+nfull.recall)
np.mean(nfull.f1_Score)

?pd.DataFrame.merge
