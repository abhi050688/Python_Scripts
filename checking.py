# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 10:18:47 2018

@author: Abhishek S
"""

def data_creation(train,test_size=0):
    train['Phrase']=train.Phrase.str.replace('-',' ')
    train['phrase_in_int']=train['Phrase'].apply(word_to_int)
    sentenceid=np.array(set(train.phrase_in_int.values))
    trn=int(len(sentenceid)*(1-test_size))
    train_data=train.loc[train.SentenceId.isin(sentenceid[:trn]),:]
    test_data=test.loc[train.SentenceId.isin(sentenceid[trn:]),:]
    return train_data,test_data


trn_data,test_data=data_creation(train,0.3)    
