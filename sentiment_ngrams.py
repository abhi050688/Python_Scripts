# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 17:44:34 2018

@author: Abhishek S
"""

import os
import pandas as pd
import numpy as np
from random import shuffle
import nltk
from nltk.util import ngrams
from collections import Counter
from keras.layers import Dense,Embedding,Activation,Dropout,Input,LSTM,Bidirectional
from keras.models import Model
import keras.backend as k
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.utils import to_categorical

os.chdir('E:/Analysis/Movie-setiment')
os.listdir()
train=pd.read_csv("train.tsv",delimiter='\t')
train.head()
def split_data(train,test_size=0):
    np.random.seed(101)
    sentenceid=np.array(list(set(train.SentenceId.values)))
    shuffle(sentenceid)
    trn=int(len(sentenceid)*(1-test_size))
    train_data=train.loc[train.SentenceId.isin(sentenceid[:trn]),:]
    test_data=train.loc[train.SentenceId.isin(sentenceid[trn:]),:]
    return train_data,test_data
train,test=split_data(train,test_size=0.3)
train.head()
splt=True
def create_ngrams(string,n):
    tokens=nltk.word_tokenize(string)
    ng=ngrams(tokens,n)
    return list(ng)


def processing_data(train,ngrams,vocab=None,splt=False):
    phrases=train.Phrase.str.lower()
    if splt:
        phrases=phrases.str.replace('-'," ")
    phrase_tokens=phrases.apply(create_ngrams,n=ngrams)
    if vocab is None:
        vocab={}
        c=1
        for i in phrase_tokens:
            for j in i:
               if j not in vocab:
                   vocab[j]=c
                   c+=1
        vocab['unk']=c
    phrase_tokens_int=phrase_tokens.apply(lambda x:[vocab[i]  if i in vocab.keys() else vocab['unk'] for i in x])
    return phrase_tokens_int,vocab

train_x,vocab=processing_data(train,ngrams=2)
test_x,_=processing_data(test,ngrams=2,vocab=vocab)
train_x.head()


train['x']=train_x
test['x']=test_x
train['ln']=train['x'].apply(lambda y: len(y))
test['ln']=test['x'].apply(lambda y: len(y))

trn=train.loc[train['ln']>0,:]
tst=test.loc[test['ln']>0,:]
allgrams=[]
for i in trn.x:
    allgrams+=i
allgrams_count=Counter(allgrams)
allgrams_count=dict(allgrams_count)
small_n=[]
lngth=5
for i,j in allgrams_count.items():
    if j <lngth:
        small_n.append(i)
len(set(small_n))

new_vocab=vocab.copy()

for i,j in new_vocab.keys():
    if allgrams_count[j]<lngth:
        new_vocab[i]=new_vocab['unk']



rev_vocab=dict(zip(vocab.values(), vocab.keys()))

trn_x=trn['x'].apply(lambda x:np.array(x))
tst_x=tst['x'].apply(lambda x:np.array(x))
trn.head()
tst_x.head()
x=trn_x.values
y=to_categorical(trn.Sentiment,num_classes=5)
y_test=to_categorical(tst.Sentiment,num_classes=5)
length=len(max(train_x,key=len))
padded_x=pad_sequences(x,maxlen=length)
padded_test_x=pad_sequences(tst_x.values,maxlen=length)
k.clear_session()
review=Input(shape=(length,))
embedding=Embedding(input_dim=len(vocab)+1,output_dim=30,input_length=length)(review)
X=Dropout(0.1)(embedding)
X=Bidirectional(LSTM(32,return_sequences=True,kernel_regularizer=regularizers.l2(0.1)))(X)
X=Dropout(0.1)(X)
X=LSTM(32,return_sequences=False,kernel_regularizer=regularizers.l2(.2))(X)
X=Dense(5)(X)
X=Activation('softmax')(X)
model=Model(inputs=review,outputs=X)
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
#model.fit(x=padded_x,y=y,batch_size=512,validation_data=(padded_test_x,y_test),epochs=20)
7
