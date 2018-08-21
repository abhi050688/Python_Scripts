# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:06:27 2017

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import sframe as sf
import json
from math import log
from math import exp
import matplotlib.pyplot as plt
loans=sf.SFrame('E:/Machine learning Classification/Week 5/lending-club-data.gl')
loans=loans.to_dataframe()
loans.head()
fl=open('E:/Machine learning Classification/Week 5/module-8-assignment-2-train-idx.json')
train_idx=json.load(fl)
fl.close()
fl=open('E:/Machine learning Classification/Week 5/module-8-assignment-2-test-idx.json')
test_idx=json.load(fl)
fl.close()

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target='safe_loans'
dt={0:1,1:-1}
loans[target]=loans['bad_loans'].map(dt)

del loans['bad_loans']
loans=loans[[target]+features]
for feat in features:
    one_hot=pd.get_dummies(loans[feat],prefix=feat)
    loans=pd.concat([loans,one_hot],axis='columns')
    del loans[feat]
loans.head()
train_data=loans.iloc[train_idx,:]
test_data=loans.iloc[test_idx,:]

def intermediate_node_weighted_mistakes(labels_in_node,data_weights):
    wm_p=sum(data_weights[labels_in_node==1])
    wm_n=sum(data_weights[labels_in_node==-1])
    if wm_p<wm_n:
        return(-1,wm_p)
    else:
        return(1,wm_n)
example_labels = pd.Series([-1, -1, 1, 1, 1])
example_data_weights = pd.Series([1., 2., .5, 1., 1.])
if intermediate_node_weighted_mistakes(example_labels, example_data_weights) == (-1, 2.5):
    print 'Test passed!'
else:
    print 'Test failed... try again!'

def best_splitting_feature(data,features,weights,target):
    best_feature=str()
    best_err=1e10
    data['wts']=weights
    for feat in features:
        minorL,errL=intermediate_node_weighted_mistakes(data.loc[data[feat]==0,target],data.loc[data[feat]==0,'wts'])
        minorR,errR=intermediate_node_weighted_mistakes(data.loc[data[feat]==1,target],data.loc[data[feat]==1,'wts'])
        total=errL+errR
        if total<best_err:
            best_err=total
            best_feature=feat
    return(best_feature)

def create_leaf(labels,weights):
    leaf={'is_leaf':True,
          'left':None,
          'right':None,
          'splitting_feature':None}
    m,err=intermediate_node_weighted_mistakes(labels,weights)
    leaf['prediction']=m
    return(leaf)

def weighted_decision_tree_create(data,features,target,data_weights,current_depth=1,max_depth=10):
    remaining_features=features[:]
    minor,err=intermediate_node_weighted_mistakes(data[target],data_weights)
    if err==0:
        print('First condition reached at %s depth and %s obs'%(current_depth,len(data)))
        return(create_leaf(data[target],data_weights))
    if len(remaining_features)==0:
        print('Second Condition: No remaining features at depth %s and %s obs'%(current_depth,len(data)))
        return(create_leaf(data[target],data_weights))
    if current_depth>max_depth:
        print('Third Condition: Max depth reached at %s depth and %s obs'%(current_depth,len(data)))
        return(create_leaf(data[target],data_weights))
    best_feature=best_splitting_feature(data.copy(),remaining_features,data_weights,target)
    left_data=data[data[best_feature]==0]
    left_wts=data_weights[data[best_feature]==0]
    right_data=data[data[best_feature]==1]
    right_wts=data_weights[data[best_feature]==1]
    if len(left_data)==len(data):
        print('Fourth Condition Left: perfect split at %s depth and %s obs'%(current_depth,len(data)))
        return(create_leaf(left_data[target],left_wts))
    if len(right_data)==len(data):
        print('Fourth Condition Right: perfect split at %s depth and %s obs'%(current_depth,len(right_data)))
        return(create_leaf(right_data[target],right_wts))
    remaining_features.remove(best_feature)
    left=weighted_decision_tree_create(left_data,remaining_features,target,left_wts,current_depth+1,max_depth)
    right=weighted_decision_tree_create(right_data,remaining_features,target,right_wts,current_depth+1,max_depth)
    return({'is_leaf':False,
            'left':left,
            'right':right,
            'prediction':None,
            'splitting_feature':best_feature})

def count_trees(tree):
    if tree['is_leaf']:
        return(1)
    else:
        return(1+count_trees(tree['left'])+count_trees(tree['right']))


def classify(tree,x,annotate=False):
#    print(x)
    if tree['is_leaf']:
        return(tree['prediction'])
#    print(tree['splitting_feature'])
    if x[tree['splitting_feature']]==0:
        if annotate:
            print('the splitting feature is %s and turned left'%(tree['splitting_feature']))
        return(classify(tree['left'],x,annotate))
    else:
        if annotate:
            print('the splitting feature is %s and turned right'%(tree['splitting_feature']))
        return(classify(tree['right'],x,annotate))

def evaluate_classification_error(tree,data):
    prediction=data.apply(lambda x:classify(tree,x),axis='columns')
    err=sum(prediction!=data['safe_loans'])
    return(err/float(len(data)))

wts=pd.Series(np.repeat(1.,10)).append(pd.Series(np.repeat(0.,len(train_data)-20))).append(pd.Series(np.repeat(1.,10)))
wts.reset_index(inplace=True,drop=True)

train_data.reset_index(inplace=True,drop=True)
train_data.index
wts.index
features=list(train_data.columns)
features.remove(target)
small_data_decision_tree_subset_20=weighted_decision_tree_create(train_data,features,target,wts,max_depth=2)
subset20=train_data.head(10).append(train_data.tail(10))
evaluate_classification_error(small_data_decision_tree_subset_20,train_data)
evaluate_classification_error(small_data_decision_tree_subset_20,subset20)


def adaboost_with_tree_stumps(data,features,target,num_tree):
    weights=list()
    trees=[]
    wts=pd.Series(np.repeat(1.,len(train_data)))
    for i in xrange(num_tree):
        tree=weighted_decision_tree_create(data.copy(),features,target,wts.copy(),max_depth=1)
        prediction=data.apply(lambda x:classify(tree,x),axis='columns')
        weighted_err=sum(wts[data[target]!=prediction])/sum(wts)
        weight=0.5*(np.log((1-weighted_err)/weighted_err))
        weights.append(weight)
        trees.append(tree)
        correct=prediction==data[target]
        adjustment=correct.apply(lambda x:exp(-weight) if x else exp(weight))
        wts=wts*adjustment
        wts=wts/wts.sum()
    return weights,trees

def predict_adaboost(weights,trees,data):
    sn=pd.Series(np.repeat(0.,len(data)))
    for i,tree in enumerate(trees):
        prediction=data.apply(lambda x:classify(tree,x),axis='columns')
        sn=sn+weights[i]*prediction
    return(sn.apply(lambda y:1 if y>0 else -1))

wts ,adb=adaboost_with_tree_stumps(train_data,features,target,num_tree=10)

plt.plot(wts)
wts30,adb30=adaboost_with_tree_stumps(train_data,features,target,num_tree=30)
error_all = []
for n in xrange(1, 31):
    predictions = predict_adaboost(wts30[:n], adb30[:n], train_data)
    error = sum(train_data[target]!=predictions)/float(len(train_data))
    error_all.append(error)
    print "Iteration %s, training error = %s" % (n, error_all[n-1])


example_data_weights = pd.Series(len(train_data)* [1.5])
if best_splitting_feature(train_data.copy(), features, example_data_weights, target) == 'term. 36 months':
    print 'Test passed!'
else:
    print 'Test failed... try again!'

example_data_weights = pd.Series([1.0 for i in range(len(train_data))])
small_data_decision_tree = weighted_decision_tree_create(train_data, features, target,
                                        example_data_weights, max_depth=2)
if count_trees(small_data_decision_tree) == 7:
    print 'Test passed!'
else:
    print 'Test failed... try again!'
    print 'Number of nodes found:', count_trees(small_data_decision_tree)
    print 'Number of nodes that should be there: 7' 

stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree=2)
def print_stump(tree):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('_')
    print '                       root'
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]{1}[{0} == 1]    '.format(split_name, ' '*(27-len(split_name)))
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                 (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))
print_stump(tree_stumps[1])

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), error_all, '-', linewidth=4.0, label='Training error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.legend(loc='best', prop={'size':15})

plt.rcParams.update({'font.size': 16})
test_data.reset_index(inplace=True,drop=True)
test_err=[]
for i in xrange(1,31):
    prediction=predict_adaboost(wts30[:i],adb30[:i],test_data)
    err=sum(prediction!=test_data[target])/float(len(test_data))
    test_err.append(err)
plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), error_all, '-', linewidth=4.0, label='Training error')
plt.plot(range(1,31), test_err, '-', linewidth=4.0, label='Test error')

plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.rcParams.update({'font.size': 16})
plt.legend(loc='best', prop={'size':15})
plt.tight_layout()























