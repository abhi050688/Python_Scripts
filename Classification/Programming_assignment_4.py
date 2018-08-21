# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:31:37 2017

@author: Abhishek S
"""

filename='E:/Machine learning Classification/Week 4/lending-club-data.csv'
trn='E:/Machine learning Classification/Week 4/module-6-assignment-train-idx.json'
tst='E:/Machine learning Classification/Week 4/module-6-assignment-validation-idx.json'

import sframe as sf
import pandas as pd
import numpy as np
import json

lending=pd.read_csv(filename)
lending.head()
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
lending['bad_loans'].describe()
lending['bad_loans'].unique()
dt={0:1,1:-1}
lending[target]=lending['bad_loans'].map(dt)
lending[[target,'bad_loans']].head(10)
del lending['bad_loans']
#?pd.get_dummies
lending=lending[[target]+features]
n_d=lending[features]
for feat in features:
    n_d=pd.concat([n_d,pd.get_dummies(n_d[feat],prefix=feat)],axis='columns')
    del n_d[feat]
n_d[target]=lending[target]
fl=open(trn)
train_idx=json.load(fl)
fl.close()
fl=open(tst)
test_idx=json.load(fl)
fl.close()
train=n_d.iloc[train_idx,:]
test=n_d.iloc[test_idx,:]
train.groupby(target)[target].count()
test.groupby(target)[target].count()

def reached_min_node(series,min_size):
    if len(series)<=min_size:
        return(True)
    else:
        return(False)
def err_reduction(error_before,error_after):
    return(error_before-error_after)

def intermediate_num_mist(ylabels):
    if len(ylabels)==0:return(0)
    if sum(ylabels)>=0:
        return(sum(ylabels<0))
    else:
        return(sum(ylabels>0))
def best_splitting_feat(data,target,features):
    best_feat=str()
    best_err=1e7
    for feat in features:
        err_r=intermediate_num_mist(data.loc[data[feat]==1,target])
        err_l=intermediate_num_mist(data.loc[data[feat]==0,target])
        err_t=err_r+err_l
        if err_t<best_err:
            best_feat=feat
            best_err=err_t
    return(best_feat)


def create_leaf(ylabels):
    leaf={'is_leaf':True,
          'right':None,
          'left':None,
          'splitting_feature':None
            }
    if sum(ylabels)>=0:
        leaf['prediction']=1
    else:
        leaf['prediction']=-1
    return(leaf)

def decision_tree_create(data,target,features,current_depth=1,max_depth=6,min_node_size=1,min_err_redux=0.0):
    remaining_features=features[:]
    ylabels=data[target]
    err_b=intermediate_num_mist(ylabels)/float(len(ylabels))
    if intermediate_num_mist(ylabels) ==0:
        print("First condition satisfied at depth %s with node size %s ."%(current_depth,len(ylabels)))
        return(create_leaf(ylabels))
    if len(remaining_features)==0:
        print("Second COndition satisfied at %s depth and %s node size"%(current_depth,len(ylabels)))
        return(create_leaf(ylabels))
    if reached_min_node(ylabels,min_node_size):
        print("Reached mini node size at %s depth and %s node size"%(current_depth,len(ylabels)))
        return(create_leaf(ylabels))
    if current_depth>=max_depth:
        print("Reached maximum depth at %s depth and %s node size"%(current_depth,len(ylabels)))
        return(create_leaf(ylabels))
    best_feature=best_splitting_feat(data,target,remaining_features)
    print('best_feat %s'%(best_feature))
    left_data=data.loc[data[best_feature]==0,:]
    right_data=data.loc[data[best_feature]==1,:]
    err_a_l=intermediate_num_mist(left_data[target])
    err_a_r=intermediate_num_mist(right_data[target])
    err_a_t=(err_a_l+err_a_r)/float(len(ylabels))
    print('err_b',err_b,'err_a',err_a_t)
    if err_reduction(err_b,err_a_t) <=min_err_redux:
        print("Minimum error reduction size reached %f  at %s depth with %s node size "%(err_reduction(err_b,err_a_t),current_depth,len(ylabels)))
        return(create_leaf(ylabels))
    if len(left_data)==len(data):
        print("All at same left split %s depth with %s node size"%(current_depth,len(ylabels)))
        return(create_leaf(left_data[target]))
    if len(right_data)==len(data):
        print("All same split at Right %s depth with %s node size "%(current_depth,len(ylabels)))
        return(create_leaf(right_data[target]))
    remaining_features.remove(best_feature)
    left_split=decision_tree_create(left_data,target,remaining_features,current_depth+1,max_depth,min_node_size,min_err_redux)
    right_split=decision_tree_create(right_data,target,remaining_features,current_depth+1,max_depth,min_node_size,min_err_redux)
    return({'is_leaf':False,
            'splitting_feature':best_feature,
            'left':left_split,
            'right':right_split,
            'prediction':None})

max_depth=6
min_node_size=100
min_error_reduction=0.0
featrs=list(train.columns)
featrs.remove(target)
my_decision_tree_new=decision_tree_create(train,target,featrs,max_depth=6,min_node_size=100,min_err_redux=0.0)    

my_decision_tree_old=decision_tree_create(train,target,featrs,max_depth=6,min_node_size=0,min_err_redux=-1)

def classify(tree,x,annotate=False):
    if tree['is_leaf']:
        if annotate:
            print('It is a leaf')
        return(tree['prediction'])
    if x[tree['splitting_feature']]==1:
        if annotate:
            print('Moving right %s Splitting '%(tree['splitting_feature']))
        return(classify(tree['right'],x,annotate))
    else:
        if annotate:
            print('Moving left %s Splitting'%(tree['splitting_feature']))
        return(classify(tree['left'],x,annotate))

print test.iloc[0,:]
print 'Predicted class: %s ' % classify(my_decision_tree_new, test.iloc[0,:])
classify(my_decision_tree_new, test.iloc[0,:], annotate = True)
classify(my_decision_tree_old, test.iloc[0,:], annotate = True)

def evaluate_classification_error(tree,data):
    data['prediction']=data.apply(lambda x:classify(tree,x),axis='columns')
    err=sum(data['prediction']!=data['safe_loans'])
    return(err/float(len(data)))

print(evaluate_classification_error(my_decision_tree_new,test))
print(evaluate_classification_error(my_decision_tree_old,test))

model_1=decision_tree_create(train,target,featrs,current_depth=1,max_depth=2,min_node_size=0,min_err_redux=-1)
model_2=decision_tree_create(train,target,featrs,current_depth=1,max_depth=6,min_node_size=0,min_err_redux=-1)
model_3=decision_tree_create(train,target,featrs,current_depth=1,max_depth=14,min_node_size=0,min_err_redux=-1)

print "Training data, classification error (model 1):", evaluate_classification_error(model_1, test)
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, test)
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, test)

print "Training data, classification error (model 1):", evaluate_classification_error(model_1, train)
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, train)
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, train)


def count_tree(tree):
    if tree['is_leaf']:
        return(1)
    else:
        return(count_tree(tree['left'])+count_tree(tree['right']))
print(count_tree(model_1))
print(count_tree(model_2))
print(count_tree(model_3))


model_4=decision_tree_create(train,target,featrs,current_depth=1,max_depth=6,min_node_size=0,min_err_redux=-1)
model_5=decision_tree_create(train,target,featrs,current_depth=1,max_depth=6,min_node_size=0,min_err_redux=0)
model_6=decision_tree_create(train,target,featrs,current_depth=1,max_depth=6,min_node_size=0,min_err_redux=5)
validation_set=test
print(count_tree(model_4))
print(count_tree(model_5))
print(count_tree(model_6))

print "Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_set)
print "Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_set)
print "Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_set)

model_7=decision_tree_create(train,target,featrs,current_depth=1,max_depth=6,min_node_size=0,min_err_redux=-1)
model_8=decision_tree_create(train,target,featrs,current_depth=1,max_depth=6,min_node_size=2000,min_err_redux=-1)
model_9=decision_tree_create(train,target,featrs,current_depth=1,max_depth=6,min_node_size=50000,min_err_redux=-1)


print(count_tree(model_7))
print(count_tree(model_8))
print(count_tree(model_9))

print "Validation data, classification error (model 4):", evaluate_classification_error(model_7, validation_set)
print "Validation data, classification error (model 5):", evaluate_classification_error(model_8, validation_set)
print "Validation data, classification error (model 6):", evaluate_classification_error(model_9, validation_set)




