# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:30:18 2018

@author: Abhishek S
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 18:57:33 2018

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

#os.chdir('E:/Analysis/Movie-setiment')
trn_data=pd.read_csv("..input/train.tsv",delimiter='\t')
test=pd.read_csv("../input/test.tsv",delimiter='\t')
trn_data.Phrase=trn_data.Phrase.str.lower()
a=trn_data.Phrase.values
a=list(a.flatten())
a=nltk.word_tokenize(' '.join(a))
a=list(set(a))
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
trn_x,_=convert_int(trn_data)
test_x,_=convert_int(test)
trn_y=trn_data.Sentiment
length=55
b_padded=pad_sequences(trn_x,maxlen=length,dtype='float32')
y=to_categorical(trn_y,num_classes=5)
b_padded_test=pad_sequences(test_x,maxlen=length,dtype='float32')
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
model.fit(x=b_padded,y=y,epochs=7,batch_size=512)


prediction=model.predict(pad_sequences(b_padded_test,maxlen=length,dtype='float32'))
predict=np.argmax(prediction,axis=1)
test['Sentiment']=predict
test[['PhraseId','Sentiment']].to_csv('predictions_2.csv',index=False)
