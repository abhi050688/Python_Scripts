import pandas as pd
import numpy as np
import os
import pickle
from keras.models import Model
from keras.layers import Embedding,Dense,Dropout,Activation,LSTM,Input,Bidirectional,TimeDistributed,concatenate
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
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import regularizers

os.chdir('E:/Analysis/Movie-setiment')
train=pd.read_csv("train.tsv",delimiter='\t')
train.head()

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

trn,test=split_data(train,test_size=0.3)
trn.head()
trn=trn.loc[trn.Phrase!=" ",:]
trn.reset_index(drop=True,inplace=True)
mn_sent=trn.groupby('SentenceId')['Sentiment'].mean()
plt.hist(mn_sent,bins=30)

tfidf=TfidfVectorizer()
trn_x=tfidf.fit_transform(trn.Phrase.str.lower())
trn_y=trn.Sentiment.values
trn_x.shape
test_x=tfidf.transform(test.Phrase.str.lower())
test_y=test.Sentiment.values
trn_y=to_categorical(trn_y,num_classes=5)
test_y=to_categorical(test_y,num_classes=5)
k.clear_session()
rev=Input(shape=(trn_x.shape[1],),dtype='float32')
X=Dense(units=32,kernel_regularizer=regularizers.l2(1e-3))(rev)
X=Activation('relu')(X)
X=Dropout(0.1)(X)
X=Dense(32,kernel_regularizer=regularizers.l2(1e-3))(X)
X=Activation('relu')(X)
X=Dropout(0.1)(X)
X=Dense(5,kernel_regularizer=regularizers.l2(1e-3))(X)
X=Activation('softmax')(X)
model=Model(inputs=rev,outputs=X)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.summary()
model.fit(x=trn_x,y=trn_y,batch_size=128,epochs=20,validation_data=(test_x,test_y))
model.save(filepath="/tfidf.h5")


trn['length']=trn.Phrase.apply(lambda x:np.float32(len(x.split(' '))))
length=trn.length.values
test['length']=test.Phrase.apply(lambda x:np.float32(len(x.split(' '))))
tst_length=test.length.values



k.clear_session()
rev=Input(shape=(trn_x.shape[1],),dtype='float32')
X=Dense(units=32,kernel_regularizer=regularizers.l2(1e-3))(rev)
X=Activation('relu')(X)
X=Dropout(0.1)(X)
aux_input=Input(shape=(1,),dtype='float32')
X=Dense(32,kernel_regularizer=regularizers.l2(1e-3))(X)
X=Activation('relu')(X)
X=Dropout(0.1)(X)
X=concatenate([X,aux_input])
X=Dense(16,kernel_regularizer=regularizers.l2(1e-3))(X)
X=Activation('relu')(X)
X=Dropout(0.1)(X)
X=Dense(5,kernel_regularizer=regularizers.l2(1e-3))(X)
X=Activation('softmax')(X)
model=Model(inputs=[rev,aux_input],outputs=X)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.summary()
model.fit(x=[trn_x,length],y=trn_y,batch_size=128,epochs=20,validation_data=([test_x,tst_length],test_y))
predictions=model.predict(x=[test_x,tst_length])
predictions=np.argmax(predictions,axis=1)
pd.DataFrame(predictions,columns=['predictions']).to_csv('tfidf_length.csv')
test_data=test.copy()
test_data['predictions']=predictions
test_data.head(20)
test_data.loc[test_data.predictions>3,:].shape
test_data.groupby('Sentiment')['Sentiment'].count()
test_data.pi
test_data.pivot_table(values='Phrase',index='Sentiment',columns='predictions',aggfunc='count',fill_value=0)



