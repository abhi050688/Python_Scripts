# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 22:37:56 2018

@author: Abhishek S
"""

import numpy as np
import pandas as pd
import os
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense,Activation,Input,Dropout,concatenate
import keras.regularizers as reg
from keras.models import Model
import keras.backend as k
from keras.utils import to_categorical
from sklearn.metrics import classification_report,confusion_matrix

   


os.chdir('E:/Analysis/Movie-setiment')
train=pd.read_csv("train.tsv",delimiter='\t')
train.head()

def split_data(train,test_size=0,seed=101):
    np.random.seed(seed)
    sentenceid=np.array(list(set(train.SentenceId.values)))
    shuffle(sentenceid)
    trn=int(len(sentenceid)*(1-test_size))
    train_data=train.loc[train.SentenceId.isin(sentenceid[:trn]),:]
    test_data=train.loc[train.SentenceId.isin(sentenceid[trn:]),:]
    return train_data,test_data

train['Phrase']=train.Phrase.str.lower()
trn,tst=split_data(train,test_size=0.3)
trn=trn.loc[trn.Phrase!=" ",:]
phrase_trn=trn.Phrase.values
phrase_tst=tst.Phrase.values

min_df=20

tfidf=TfidfVectorizer(token_pattern=r"\b[\'a-z]+\b",min_df=min_df)
x_trnb=tfidf.fit_transform(phrase_trn)
len(tfidf.vocabulary_)
y_trn=trn.Sentiment.values

x_tst=tfidf.transform(phrase_tst)
y_tst=tst.Sentiment.values
x_tst.shape

y_trn=to_categorical(y_trn,num_classes=5)
y_tst=to_categorical(y_tst,num_classes=5)
clss={0:1.1,1:1.05,2:1,3:1.05,4:1.1}
k.clear_session()
review=Input(dtype="float32",shape=(x_trnb.shape[1],))
X=Dense(128,activation='tanh',kernel_regularizer=reg.l2(l=1e-3))(review)
X=Dropout(0.2)(X)
X=Dense(64,activation='relu',kernel_regularizer=reg.l2(l=1e-4))(X)
X=Dropout(0.05)(X)
X=Dense(32,activation='relu',kernel_regularizer=reg.l2(l=1e-5))(X)
X=Dense(5,activation='softmax')(X)
model=Model(inputs=review,outputs=X)
model.compile(optimizer='adam',metrics=['categorical_accuracy'],loss='categorical_crossentropy')
model.summary()
model.fit(x=x_trnb,y=y_trn,batch_size=512,epochs=10,validation_data=(x_tst,y_tst),class_weight=clss)
#model.fit(x=x_trnb,y=y_trn,batch_size=512,epochs=5,validation_data=(x_tst,y_tst))
model.evaluate(x=x_tst,y=y_tst)

prediction=model.predict(x_tst)
predict_class=np.argmax(prediction,axis=1)
rp=classification_report(tst.Sentiment.values,predict_class)
cm=confusion_matrix(tst.Sentiment.values,predict_class)
print(rp)
print(cm)
tst.groupby('Sentiment')['Sentiment'].count()

length=trn.Phrase.apply(lambda x:np.float32(len(x.split(' '))))
length_tst=tst.Phrase.apply(lambda x:np.float32(len(x.split(' '))))

clss={0:1.1,1:1.05,2:1,3:1.05,4:1.1}
k.clear_session()
review=Input(dtype="float32",shape=(x_trnb.shape[1],))
X=Dense(128,activation='tanh',kernel_regularizer=reg.l2(l=1e-3))(review)
X=Dropout(0.2)(X)
X=Dense(64,activation='relu',kernel_regularizer=reg.l2(l=1e-4))(X)
X=Dropout(0.05)(X)
X=Dense(32,activation='relu',kernel_regularizer=reg.l2(l=1e-5))(X)
lngth=Input(shape=(1,),dtype='float32')
X=concatenate([X,lngth])
X=Dense(5,activation='softmax')(X)
model=Model(inputs=[review,lngth],outputs=X)
model.compile(optimizer='adam',metrics=['categorical_accuracy'],loss='categorical_crossentropy')
model.summary()
#model.fit(x=[x_trnb,length],y=y_trn,batch_size=512,epochs=5,validation_data=([x_tst,length_tst],y_tst),class_weight=clss)
model.fit(x=[x_trnb,length],y=y_trn,batch_size=512,epochs=10,validation_data=([x_tst,length_tst],y_tst))


