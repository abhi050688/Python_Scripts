# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:47:50 2017

@author: Abhishek S
"""

import pandas as pd
import numpy as np
from pandas import DataFrame as df
import json
import sframe as sf

products=pd.read_csv('E:/Machine learning Classification/Week 2/amazon_baby_subset.csv')
products.head()
products.dtypes

#Get crossproduct b/w sentiment and ratings?
products.describe()
products.isnull().sum()
products.info()
products.count() #<- Count of non missing values only
len(products.index)
?df.g
idx=products[products['review'].isnull()].index
products.iloc[idx,:]
products['review'].fillna(value='',inplace=True)

def clean_review(text):
    import string
    return(text.translate(None,string.punctuation))

products['review_clean']=products['review'].apply(clean_review)

wrd=open('E:/Machine learning Classification/Week 2/important_words.json')
imp_word=json.load(wrd)
wrd.close()

for words in imp_word:
    products[words]=products['review_clean'].apply(lambda x: x.split().count(words))
?df.to_xarray
products.head()
(lp,ll)=products[products['perfect']>0].shape #<- 2955
?df.as_matrix

features=[]
label=str()
products['perfect'].as_matrix()

def get_numpy_data(dataframe,features,label):
    dataframe['constant']=1
    features=['constant']+features
    feature_matrix=dataframe[features].as_matrix()
    out=dataframe[label].as_matrix()
    return(feature_matrix,out)

features=['perfect','baby']
label='sentiment'
tr,tr_target=get_numpy_data(products,features,label)


def predict_probability(feature_matrix,coeff):
    score=np.dot(feature_matrix,coeff)
    p=1/(1+np.exp(-score))
    return(p)
coeff=np.array([1,2,3])
?np.apply_over_axes
x=predict_probability(tr,coeff)
x.apply_along_axis(lambda x: x[0] )
x=pd.DataFrame(x)
sent=(tr_target==-1)
sent=abs(sent-x)
sent.sum()
product(sent)
np.product(sent)
np.min(sent)
def logliklihood(feature_matrix,coeff,target):
    score=np.dot(feature_matrix,coeff)
    sentiment=(target==1)
    ll=np.sum(score*(sentiment-1)-np.log(1.+np.exp(-score)))
    return(ll)

def feature_derivative(feature_matrix,coeff,target):
    probability=predict_probability(feature_matrix,coeff)
    sentiment=(target==1)
    error=(sentiment-probability)
    return(np.dot(np.transpose(feature_matrix),error))


def gradient_ascent(dataframe,features,label,initial_wts,eta,itera=1000):
    feature_matrix,target=get_numpy_data(dataframe,features,label)
    i=0
    while(i<itera):
        ll=logliklihood(feature_matrix,initial_wts,target)
        print 'Loglikelihood at ',i,' ',ll 
        feature_der=feature_derivative(feature_matrix,initial_wts,target)
        for j in xrange(len(initial_wts)):
            initial_wts[j]=initial_wts[j]+eta*feature_der[j]
        i=i+1
    return(initial_wts)

features=imp_word
dataframe=products
label='sentiment'
initial_wts=np.repeat(0.,len(imp_word)+1)
eta=1e-7
itera=301

coeff=gradient_ascent(dataframe,features,label,initial_wts.copy(),eta,itera)
scores=np.dot(feature_matrix,coeff)
sentiment=sum(scores>=0)
correct=sum((scores>=0)==(target==1)) #<-39903
accuracy=float(correct)/len(target)
coeff_wo_i=coeff[1:]
word_coeff=pd.DataFrame({'words':imp_word,'coeff':coeff_wo_i })
word_coeff.sort_values(by='coeff',ascending=False,inplace=True)

flt=open('E:/Machine learning Classification/Week 2/module-4-assignment-train-idx.json')
trn_idx=json.load(flt)
flt.close()

flt=open('E:/Machine learning Classification/Week 2/module-4-assignment-validation-idx.json')
tst_idx=json.load(flt)
flt.close()

train=products.iloc[trn_idx,:]
test=products.iloc[tst_idx,:]

feature_mat_train,target_train=get_numpy_data(train,features,label)
feature_mat_test,target_test=get_numpy_data(test,features,label)

def likelihood_l2(feature_matrix,target,coeff,l2_penalty):
    score=np.dot(feature_matrix,coeff)
    ll=np.sum(score*((target==1)-1)-np.log(1+np.exp(-score))) - l2_penalty*np.sum(pow(coeff[1:],2))
    return(ll)

def feature_der_l2(feature_matrix,target,coeff):
    probability=predict_probability(feature_matrix,coeff)
    errors=(target==1)-probability
    np.dot(np.transpose(feature_matrix),errors)

def gradient_ascent_2(dataframe,features,label,initial_wts,eta,l2_penalty,itera=1000):
    feature_matrix,target=get_numpy_data(dataframe,features,label)
    i=0
    while(i<itera):
        feature_der=feature_derivative(feature_matrix,initial_wts,target)
        ll=likelihood_l2(feature_matrix,target,initial_wts,l2_penalty)
        #print 'LogLikelihood at i= ',i,' ',ll
        for j in xrange(len(initial_wts)):
#            print 'initial_wts ',j,' ',initial_wts[j]
            if(j==0):
                initial_wts[j]=initial_wts[j]+eta*feature_der[j]
            else:
                initial_wts[j]=initial_wts[j]*(1-2*eta*l2_penalty)+eta*feature_der[j]
#            print 'after change ',j,' ',initial_wts[j]
        i=i+1
    return(initial_wts)

dataframe=train
l2_penalty=[0,4,10,1e2,1e3,1e5]
itera=501
initial_wts=np.repeat(0.,len(imp_word)+1)
features=imp_word
label='sentiment'
eta=5e-6
#coeff=gradient_ascent_2(train,features,label,initial_wts.copy(),eta,l2_penalty[5],itera=10)
coeff=dict()
for k in xrange(len(l2_penalty)):
    coeff[l2_penalty[k]]=gradient_ascent_2(train,features,label,initial_wts.copy(),eta,l2_penalty[k],itera)

#coeff[l2_penalty[5]]=gradient_ascent_2(train,features,label,initial_wts.copy(),eta,l2_penalty[5],itera)

words=pd.DataFrame({'words':['intercept']+imp_word,'coeff':coeff.get(0)})
words.sort_values(by='coeff',inplace=True)
words.reset_index(inplace=True)
positive_words=list(words.iloc[189:,2])
negative_words=list(words.iloc[:5,2])
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 6

def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    
    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')
    
    table_positive_words = table[table['word'].is_in(positive_words)]
    table_negative_words = table[table['word'].is_in(negative_words)]
    del table_positive_words['word']
    del table_negative_words['word']
    
    for i in xrange(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words[i:i+1].to_numpy().flatten(),'-', label=positive_words[i], linewidth=4.0, color=color)
        
    for i in xrange(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words[i:i+1].to_numpy().flatten(),
                 '-', label=negative_words[i], linewidth=4.0, color=color)
        
    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -2, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()

coeff_new=coeff
coeff['word']=['intercept']+imp_word
table=sf.SFrame(coeff)
table=table[['word','0','4','10', '100.0', '1000.0', '100000.0']]
coeff['0']=coeff.pop(0)
coeff['100000.0']=coeff.pop(100000.0)
coeff['100.0']=coeff.pop(100.0)
coeff['1000.0']=coeff.pop(1000.0)
coeff['10']=coeff.pop(10)
coeff['4']=coeff.pop(4)

make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list=[0, 4, 10, 1e2, 1e3, 1e5])

table[table['word']=='little']
sf.SFrame.t

dataset=test
fetr,trg=get_numpy_data(dataset,features,label)
accu=dict()
i=2
for i in xrange(len(l2_penalty)-1):
    accu[l2_penalty[i]]=np.sum((np.dot(fetr,coeff[str(l2_penalty[i])]) >=0)==(trg==1))/float(len(trg))

    












0
    




a=str()
?a.translate
a=['a','b','a']
a.count('a')

