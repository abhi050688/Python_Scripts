# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 21:17:43 2018

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import os
import pickle
from keras.models import Model
from keras.layers import Embedding,Dense,Dropout,Activation,LSTM,Input,Bidirectional,TimeDistributed
import keras.backend as k
from random import shuffle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import tensorflow as tf
from sklearn.feature_extraction.text  import CountVectorizer
import nltk
import matplotlib.pyplot as plt

glove="E:/Deep Learning/Sequence Models/Week 2/mojify_w2_a2/data/glove.6B.50d.txt"

with open(glove,'r',encoding='utf8') as file:
    word2vec={}
    word_weights=np.zeros([400000,50])
    i=0
    wordlist={}
    for line in file.readlines():
        word=line.lower().strip().split(' ')
        word2vec[word[0]]=np.asarray(word[1:],dtype='float32')
        wordlist[word[0]]=i
        word_weights[i,:]=word2vec[word[0]]
        i+=1
word_weights[3,:]

fle="E:/Deep Learning/Sequence Models/Week 2/mojify_w2_a2/data/gloveobjects.file"
with open(fle,'wb') as f:
    pickle.dump((word2vec,wordlist,word_weights),f,pickle.HIGHEST_PROTOCOL)

with open(fle,'rb') as f:
    word2vec,wordlist,word_weights=pickle.load(f)

os.chdir('E:/Analysis/Movie-setiment')
train=pd.read_csv("train.tsv",delimiter='\t')
train.head()
def word_to_int(string):
    a=list()
    for j in string.lower().strip().split():
        try:
            a.append(wordlist[j])
        except KeyError as e:
            pass
    return a

def data_creation(train,test_size=0):
    train['Phrase']=train.Phrase.str.replace('-',' ')
    train['phrase_in_int']=train['Phrase'].apply(word_to_int)
    sentenceid=np.array(list(set(train.SentenceId.values)))
    trn=int(len(sentenceid)*(1-test_size))
    train_data=train.loc[train.SentenceId.isin(sentenceid[:trn]),:]
    test_data=train.loc[train.SentenceId.isin(sentenceid[trn:]),:]
    return train_data,test_data

def split_data(train,test_size=0):
    np.random.seed(101)
    sentenceid=np.array(list(set(train.SentenceId.values)))
    shuffle(sentenceid)
    trn=int(len(sentenceid)*(1-test_size))
    train_data=train.loc[train.SentenceId.isin(sentenceid[:trn]),:]
    test_data=train.loc[train.SentenceId.isin(sentenceid[trn:]),:]
    return train_data,test_data


#trn_data,test_data=data_creation(train,0.3)    
trn_data,test_data=split_data(train,0.3)
nltk.download('punkt')
trn_data.Phrase=trn_data.Phrase.str.lower()
a=trn_data.Phrase.values
a=list(a.flatten())
a=nltk.word_tokenize(' '.join(a))
a=list(set(a))
len(a)
word_vocab={j.lower():i for i,j in enumerate(a)}
word_vocab['unk']=len(word_vocab)
def convert_int(data):
    b=list()
    word_not_found=[]
    for i in data.Phrase.values:
        a=list()
        for j in i.lower().strip().split():
            try:
                a.append(word_vocab[j])
            except KeyError as e:
                word_not_found.append(e.args[0])
                a.append(word_vocab['unk'])
    #            print(e.args[0])
        b.append(a)
        a=[]
    return b,word_not_found
trn_x,wnfx=convert_int(trn_data)
test_x,wnft=convert_int(test_data)
len(set(wnfx))
trn_y=trn_data.Sentiment
test_y=test_data.Sentiment
trn_data.head()
#train_x,test_x,train_y,test_y=trn_data.phrase_in_int.values,test_data.phrase_in_int.values,trn_data.Sentiment.values,test_data.Sentiment.values
length=64
b_padded=pad_sequences(trn_x,maxlen=length,dtype='float32')
y=to_categorical(trn_y,num_classes=5)
b_padded_test=pad_sequences(test_x,maxlen=length,dtype='float32')
y_test=to_categorical(test_y,num_classes=5)
k.clear_session()
review=Input(shape=(length,),name='review',dtype='float32')
#X=Embedding(input_dim=400000,output_dim=50,input_length=length,weights=[word_weights],trainable=True)(review)
X=Embedding(input_dim=len(word_vocab)+1,output_dim=20,input_length=length,trainable=True)(review)
X=Bidirectional(LSTM(64,return_sequences=True))(X)
X=Dropout(0.2)(X)
X=Bidirectional(LSTM(64,return_sequences=False))(X)
X=Dropout(0.2)(X)
#X=TimeDistributed(Dense(5))(X)
#X=Activation('relu')(X)
X=Dense(5)(X)
X=Activation('softmax')(X)
model=Model(inputs=review,outputs=X)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.summary()
#checkpoint_path="cp.ckpt"
#checkpoint_dir=os.path.dirname(checkpoint_path)
#cp_callbacks=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=1,period=1)
#cp_callback2=tf.keras.callbacks.ModelCheckpoint('cp_2.ckpt',verbose=0)
#cp_callback3=tf.keras.callbacks.ModelCheckpoint('cp_3.ckpt',verbose=0)
#cp_callback4=tf.keras.callbacks.ModelCheckpoint('cp_4.ckpt',verbose=0)

#
#test=pd.read_csv('test.tsv',delimiter='\t')
#test.head()
#atest,_=convert_int(test)
#
#prediction=model.predict(pad_sequences(atest,maxlen=length,dtype='float32'))
#predict=np.argmax(prediction,axis=1)
#test['Sentiment']=predict
#test.head()
#test[['PhraseId','Sentiment']].to_csv('predictions.csv',index=False)
#pd.DataFrame.to
#model.save('/model.h5')
#model=load_model('/model.h5')
#?pd.DataFrame.to_csv


a=train.groupby('SentenceId')['SentenceId'].count()
plt.hist(a,bins=50,edgecolor='b')




