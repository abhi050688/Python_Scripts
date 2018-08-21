# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:24:54 2017

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import json


lending=pd.read_csv('E:/Machine learning Classification/Week 3/assignment2/lending-club-data.csv')
lending.head()
bd_sf={1:-1,0:1}
lending['safe_loans']=lending['bad_loans'].map(bd_sf)
del lending['bad_loans']

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'

fl=open('E:/Machine learning Classification/Week 3/assignment2/module-5-assignment-2-train-idx.json')
train_idx=json.load(fl)
fl.close()


fl=open('E:/Machine learning Classification/Week 3/assignment2/module-5-assignment-2-test-idx.json')
test_idx=json.load(fl)
fl.close()
lending=lending[[target]+features]
lending.head()
cat_col=lending.columns[lending.dtypes==np.object]
n_lend=lending[cat_col]
lending['v']=1
n_lend=lending.pivot(columns='grade',values='v').fillna(0)
lcopy=lending.copy()
for feat in cat_col:
    print feat
    n_lend=pd.get_dummies(lcopy[feat],prefix=feat)
    del lcopy[feat]
    lcopy=pd.concat([lcopy,n_lend],axis='columns')

del lcopy['v']
lending=lcopy.copy()


train=lending.iloc[train_idx,:]
test=lending.iloc[test_idx,:]

train.groupby(target)[target].count()/train.safe_loans.count()
test.groupby(target)[target].count()/test.safe_loans.count()

def intermediate_node_num_mistakes(y_labels):
    if len(y_labels)==0:
        return 0
    majority=int(y_labels.sum()>=0)
    if majority==0:majority=-1
    mis=sum(y_labels!=majority)
    return(mis)
    
# Test case 1
example_labels = np.array([-1, -1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 1 failed... try again!'

# Test case 2
example_labels = np.array([-1, -1, 1, 1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 3 failed... try again!'
    
# Test case 3
example_labels = np.array([-1, -1, -1, -1, -1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 3 failed... try again!'        

def best_spilliting_feature(data,features,target):
    best_feat=str()
    best_error=1e10
    for feat in features:
        left_data=data.loc[data[feat]==0,:]
        right_data=data.loc[data[feat]==1,:]
        l_mis=intermediate_node_num_mistakes(left_data[target])
        r_mis=intermediate_node_num_mistakes(right_data[target])
        tl=l_mis+r_mis
        if tl<best_error:
            best_error=tl
            best_feat=feat
    return(best_feat)
n_features=list(train.columns)
n_features.pop(0)


bf=best_spilliting_feature(train,n_features,target)


def create_leaf(y_values):
    leaf={'is_leaf':True,
          'splitting_feature':None,
          'left':None,
          'right':None
          }
    if sum(y_values)>=0:
        leaf['prediction']=1
    else:
        leaf['prediction']=-1
    return(leaf)

def decision_tree_create(data,features,target,current_depth=0,max_depth=10):
    remaining_features=features[:]
    #First condition to check
    target_labels=data[target]
    print('Subtree,depth %s (%s datapoints).' % (current_depth,len(data.index)))
    if intermediate_node_num_mistakes(target_labels)==0:
        print("First condition satisfied")
        return(create_leaf(target_labels))
    if len(remaining_features)==0:
        print('Second Condition Satisfies')
        return(create_leaf(target_labels))
    if current_depth>=max_depth:
        print('Third Condition satisfied')
        return(create_leaf(target_labels))
    best_feature=best_spilliting_feature(data,remaining_features,target)
    left_data=data.loc[data[best_feature]==0,:]
    right_data=data.loc[data[best_feature]==1,:]
    remaining_features.remove(best_feature)
    if len(left_data)==len(data):
        print('Fourth left condition')
        return(create_leaf(left_data[target]))
    if len(right_data)==len(data):
        print('Fourth_right_condition')
        return(create_leaf(right_data[target]))
    left_tree=decision_tree_create(left_data,remaining_features,target,current_depth+1,max_depth)
    right_tree=decision_tree_create(right_data,remaining_features,target,current_depth+1,max_depth)
    return({'is_leaf':False,
            'left':left_tree,
            'right':right_tree,
            'splitting_feature':best_feature,
            'prediction':None
            })
features=list(train.columns)
features.remove(target)
my_decision_tree=decision_tree_create(train,features,target,current_depth=0,max_depth=6)


def classify(tree,data,annotate=False):
    if tree['is_leaf']:
        if annotate:
            print('At leaf. Prediction= %s' % (tree['prediction']))
        return(tree['prediction'])
    else:
        if data[tree['splitting_feature']]==0:
            if annotate:
                print('At Node. Traversing Left. Spilliting feature is %s and value %s' %(tree['splitting_feature'],data[tree['splitting_feature']]))
            return(classify(tree['left'],data,annotate))
        else:
            if annotate:
                print('At Node. Traversing right. Splitting feature is %s and value %s ' %(tree['splitting_feature'],data[tree['splitting_feature']]))
            return(classify(tree['right'],data,annotate))
        

print test.iloc[0,:]
print 'Predicted class: %s ' % classify(my_decision_tree, test.iloc[0,:],True)

def evaluate_classification_error(tree,data):
    prediction=data.apply(lambda x:classify(my_decision_tree,x,True),axis=1)
    incorrect=sum(prediction!=data[target])
    return(incorrect/float(len(data)))
er=evaluate_classification_error(my_decision_tree,test)
l='left'
r='right'
print my_decision_tree[l][l][l][l][l][l]
test.reset_index(inplace=True)
test.head()
data=test.iloc[0:2,:]
prediction=data.apply(lambda x: x.safe_loans)


def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('_')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

print_stump(my_decision_tree)
print_stump(my_decision_tree[r], my_decision_tree['splitting_feature'])
print_stump(my_decision_tree[r][r], my_decision_tree[r]['splitting_feature'])
















