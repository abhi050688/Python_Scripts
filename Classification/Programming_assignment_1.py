# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:38:53 2017

@author: Abhishek S
"""

import sframe as sf
import pandas as pd
import string
import json
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model as lm
import numpy as np


amazon=pd.read_csv('E:/Machine learning Classification/amazon_baby.csv')
amazon.head()
#find nulls
##First way
sum(amazon['review'].isnull())
idxn=amazon[amazon['review'].isnull()].index
amazon.iloc[idxn,1]=''
amazon.iloc[idxn,:]

##Second way
#?pd.DataFrame.fillna
#?pd.Series.fillna
amazon['review'].fillna(value='',inplace=True)
sum(amazon['review'].isnull())
amazon.head()
amazon.iloc[idxn,:]


#Remove rating 3 and deem positive and negative review based on ratings
amazon=amazon[amazon['rating']<>3]
amazon['sentiment']=amazon['rating'].apply(lambda x: 1 if x>3 else -1 )
amazon.head(100)
amazon.dtypes


a=str()
#?a.translate
a='abhishek.sahu'
a.translate(None,string.punctuation)

def review_cleanup(text):
    return text.translate(None,string.punctuation)
amazon['review_clean']=amazon['review'].apply(review_cleanup)

json_data=open('E:/Machine learning Classification/module-2-assignment-train-idx.json')
train_idx=json.load(json_data)
json_data.close()
json_test=open('E:/Machine learning Classification/module-2-assignment-test-idx.json')
test_idx=json.load(json_test)
json_test.close()

train=amazon.iloc[train_idx,:]
test=amazon.iloc[test_idx,:]

vectorizer=CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix=vectorizer.fit_transform(train['review_clean'])
test_matrix=vectorizer.transform(test['review_clean'])
print(train_matrix)
vectorizer.vocabulary_

sentiment=lm.LogisticRegression()
sentiment.fit(X=train_matrix,y=train['sentiment'])
len(sentiment.coef_[sentiment.coef_ >=0]) #<-Quiz 1 77290
len(sentiment.coef_[0])

sample_test_data=test.iloc[10:13,:]
sample_test_data.iloc[0,1]
sample_test_data.iloc[1,:]['review_clean']
sample_test_matrix=vectorizer.transform(sample_test_data['review_clean'])
sample_predict=sentiment.predict(sample_test_matrix)
sample_score=sentiment.decision_function(sample_test_matrix)
pr=1/(1+np.exp(-sample_score))
print("%.4f" % pr[2])
sentiment.predict_proba(sample_test_matrix)




test_prob=sentiment.predict_proba(test_matrix)
test_prob[:,1]
test['probability']=test_prob[:,1]
test=test.sort_values(by='probability',axis=0,ascending=False)
test['name'].head(20)
test=test.sort_values(by='probability',axis=0,ascending=True)
test['name'].head(20)

#test['prediction']=test['probability'].apply(lambda x:1 if x>=0.5 else -1)
test['prediction']=sentiment.predict(test_matrix)

len(test[test['sentiment']==test['prediction']])
31077.0/33336

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

vectorizer_word=CountVectorizer(vocabulary=significant_words)
train_subset=vectorizer_word.fit_transform(train['review_clean'])
test_subset=vectorizer_word.transform(test['review_clean'])

simple_model=lm.LogisticRegression()
simple_model.fit(X=train_subset,y=train['sentiment'])
simple_coef_table=pd.DataFrame({'word':significant_words,'coef':simple_model.coef_.flatten()})

len(sentiment.coef_.flatten())
len(vectorizer.vocabulary_.keys())
sentiment_coef_table=pd.DataFrame({'word':vectorizer.vocabulary_.keys(),'coef':sentiment.coef_.flatten()})

sent=sentiment_coef_table[sentiment_coef_table['word'].isin(significant_words)]
sent.sort_values('coef',ascending=False)


test['simple_predict']=simple_model.predict(test_subset)
float(sum(test['sentiment']==test['prediction']))/33336
float(sum(test['sentiment']==test['simple_predict']))/33336


train['prediction']=list(sentiment.predict(train_matrix))
train['simple_predict']=simple_model.predict(train_subset)
float(sum(train['sentiment']==train['prediction']))/133416
float(sum(train['sentiment']==train['simple_predict']))/133416

float(sum(train['sentiment']==1))/133416
float(sum(test['sentiment']==1))/33336


































