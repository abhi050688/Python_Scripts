# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:45:43 2018

@author: Abhishek S
"""

import numpy as np
from keras import models,optimizers
from keras.models  import Sequential
from keras.layers import Dense,LSTM,Activation,Embedding,TimeDistributed,Dropout
from random import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.python.framework import ops
from keras import backend as k
 

file="E:/Deep Learning/Sequence Models/Week 1/Dinosorus/dinos.txt"

with open(file) as f:
    example=[line.lower() for line in f.readlines()]
vocab=list(set("".join(example)))
vocab=vocab+['strt']
len("".join(example))
len(vocab)
char_to_idx={j:i  for i,j  in enumerate(sorted(vocab))}
idx_to_char={ value:key for key,value in char_to_idx.items()}

def name_to_idx(name):
    a=[ char_to_idx[i] for i in name.strip()]
    a=[char_to_idx['strt']]+a
    return a

def idxlist_to_name(idx):
    nm=[idx_to_char[i]  for  i in idx]
    return "".join(nm).strip()

encoded=[ name_to_idx(i)  for i in example]
len(encoded)
np.random.seed(101)
shuffle(encoded)

n_examples=pad_sequences(encoded,maxlen=30,padding='post')
y=[i[1:] for i in encoded]
n_y=pad_sequences(y,maxlen=30,padding='post')
n_y=to_categorical(n_y,num_classes=len(vocab))
k.clear_session()
model=Sequential()# Input size shoule be (batch_size,30) since maxlen after padding is 30
model.add(Embedding(input_dim=len(vocab),output_dim=10,input_length=30)) # Input should become (batch_size,30,10)
model.add(LSTM(16,return_sequences=True))
model.add(LSTM(16,return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(len(vocab))))
model.add(Activation('softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.summary()
model.fit(n_examples,n_y,batch_size=30,epochs=200)

indx=char_to_idx['strt']
a=np.zeros([30])
a[0]=indx
a=a.reshape(1,30)
out=model.predict(a)
idx_to_char[np.argmax(out[:,0,:])]

def sample(names,seed=1):
    for i in range(names):
        counter=0
        indx=char_to_idx['strt']
        a=np.zeros([30])
        a=a.reshape(1,30)
        a[0,counter]=indx
        while indx!=0 and counter<30:
            out=model.predict(a)
            np.random.seed(seed+counter)
            indx=np.random.choice(np.arange(0,28),p=out[:,counter,:].ravel())
            counter+=1
            a[0,counter]=indx
        print(idxlist_to_name(a.flatten()[1:]))
        
for i in range(100):
    sample(1,10+i)



