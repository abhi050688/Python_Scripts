import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import gc

#==================================================================================================================#
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import cpu_count, pool
import pickle



pd.options.mode.chained_assignment=None
train=pd.read_csv(r"E:\Analysis\Movie-setiment\train.tsv",delimiter='\t')
a=train.loc[train.SentenceId.isin(list(range(1,1000))),:]
#a=train.copy()
a=a.loc[a.Phrase!=' ',:]

a['beg']=a['Phrase'].apply(lambda x: x.split()[0])
a['end']=a['Phrase'].apply(lambda x: x.split()[len(x.split())-1])

class tree_creation_forked:
    def __init__(self,data):
        self.a=data
        self.beg=self.a.groupby('beg').groups
        self.end=self.a.groupby('end').groups
        self.a['trees']=np.repeat(dict(),self.a.shape[0])
    
    def create_tree(self,row):
        tree=dict()
        line=row.Phrase.iat[0]
        if len(line.split())==1:
            return self.create_leaf(row)
        else:
            candidates1=self.beg[row.beg.iat[0]]
            candidates2=self.end[row.end.iat[0]]
            for j in candidates1:
                for k in candidates2:
                    if self.a.Phrase[j]+" "+self.a.Phrase[k]==line:
                        tree['left']=self.create_tree(self.a.loc[self.a.index==j,:])
                        tree['right']=self.create_tree(self.a.loc[self.a.index==k,:])
                        tree['left_word']=self.a.Phrase[j]
                        tree['right_word']=self.a.Phrase[k]
                        tree['leaf']=False
                        tree['word']=line
                        return tree
            return self.create_leaf(row)
    def create_leaf(self,row):
        line=row.Phrase.iat[0]
        tree={}
        tree['leaf']=True
        tree['word']=line
        return tree
    
    def run(self):
        i=min(self.a.index)
        row=a.loc[a.index==i,:]
        tree=self.create_tree(row)
        for i in self.a.index:
            row=a.loc[a.index==i,:]
            self.a.at[i,'trees']=self.find_tree(row,tree)
        return self.a
    
    def find_tree(self,row,tree):
        line=row.Phrase.iat[0]
        if line==tree['word']:
            return tree
        else:
            try:
                if line in tree['left_word']:
                    return self.find_tree(row,tree['left'])
                else:
                    return self.find_tree(row,tree['right'])
            except KeyError as e:
                return self.create_leaf(row)

def generate_tree(data):
    new_tree=tree_creation_forked(data)
    tree=new_tree.run()
    del new_tree
    gc.collect()
    return tree

new_tree=a.groupby('SentenceId').apply(generate_tree)
new_a=tree_creation_forked(a)
new_data=new_a.run()
new_data.iloc[22,6]




sttt="a good boy"
st2="home a good boy"
st1="the house was haunted"
st3="home house"
st4="was haunted"
print(sttt in st2)
print(st1 in st2)
print(st4 in st1)
import time
final_f=pd.DataFrame()
nsentence=train.SentenceId.nunique()
rngs=list(range(0,nsentence,100))+[nsentence]


