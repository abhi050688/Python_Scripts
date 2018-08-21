# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 09:51:42 2018

@author: Abhishek S
"""
import pandas as pd
import numpy as np
import os
import nltk


os.chdir('E:/Analysis/Movie-setiment')
train=pd.read_csv("train.tsv",delimiter='\t')
train.head()
a=train.loc[train.SentenceId==1,:]
a.reset_index(drop=True,inplace=True)
a['Phrase']=a['Phrase'].str.lower()
a['phrase_split']=a.Phrase.apply(lambda x:x.split(' '))
a['phrase_len']=a.phrase_split.apply(lambda x:len(x))
strn=' '.join(a.Phrase.values)
strn=strn.split(' ')
words=list(set(strn))
worddict={ j:i for i,j  in enumerate(words)}
word2vec=[ [ worddict[k]     for k in i     ] for i in a.phrase_split.values]
revlist={value:keys for keys,value in worddict.items()}
key=20
for i,k in enumerate(word2vec):
    rem=k[0]%key
    if  rem in hashkey:
        hashkey[rem].append(i)
    else:
        hashkey[rem]=list()
hashkey={}
dhashkey={}
for i,k in enumerate(word2vec):
    begin=k[0]
    end=k[len(k)-1]
    if  begin in hashkey:
        hashkey[begin].append(i)
    else:
        hashkey[begin]=list()
    if  end in dhashkey:
        dhashkey[end].append(i)
    else:
        dhashkey[end]=list()
revlist[12]
tree=[]
for c,i in enumerate(word2vec):
    begin,end=i[0],i[len(i)-1]
    candidates1=hashkey[begin]
    candidates2=dhashkey[end]
    fill=False
    for j in candidates1:
        for  k in candidates2:
            if word2vec[j]+word2vec[k]==i:
                tree.append([j,k])
                fill=True
    if not fill:
        tree.append([])

train['phrase_len']=train.Phrase.apply(lambda x:len(x.split(' ')))
train.shape
train.head(100)
train.loc[train.phrase_len==1.0,:].shape


