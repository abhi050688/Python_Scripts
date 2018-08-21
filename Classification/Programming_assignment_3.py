# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 16:51:52 2017

@author: Abhishek S
"""

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import json
import sframe as sf
import sklearn.tree as st
from sklearn.tree import DecisionTreeClassifier as dc

lending=pd.read_csv('E:/Machine learning Classification/Week 3/lending-club-data.csv');
lending.head()
lending.dtypes
lending['safe_loans']=lending['bad_loans'].apply(lambda x: 1 if x==0 else -1)
del lending['bad_loans']

lending['safe_loans'].describe()
?df.groupby().len
lending.groupby(by='safe_loans')['safe_loans'].size()/lending.safe_loans.count()

fl=open('E:/Machine learning Classification/Week 3/module-5-assignment-1-train-idx.json')
train_idx=json.load(fl)
fl.close()

fl=open('E:/Machine learning Classification/Week 3/module-5-assignment-1-validation-idx.json')
test_idx=json.load(fl)
fl.close()

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

lending=lending[['safe_loans']+features]
lending.dtypes
train_data=lending.iloc[train_idx,:]
test_data=lending.iloc[test_idx,:]
train_data.groupby('safe_loans')['grade'].count()/train_data.grade.count()
test_data.groupby('safe_loans')['grade'].count()/test_data.grade.count()

cat_col=list(train_data.columns[train_data.dtypes== np.object])

def on_hot_encoding(dataset,columns):
    dataset=dataset[columns]
    col=dataset.columns
    for i in col:
        series=dataset[i]
        print(i)
        a=series.unique()
        for j in a:
            print(j)
            dataset[i+'_'+j.strip()]=series.apply(lambda x:1 if x==j else 0)
    return(dataset)

dataset=train_data
n_d=on_hot_encoding(train_data,cat_col)
n_d.head()
n_d.drop(cat_col,axis='columns',inplace=True)
train_data.drop(cat_col,axis='columns',inplace=True)
train=pd.concat([train_data,n_d],axis='columns')

n_d_t=on_hot_encoding(test_data,cat_col)
n_d_t.drop(cat_col,axis='columns',inplace=True)
test_data.drop(cat_col,axis='columns',inplace=True)
valid=pd.concat([test_data,n_d_t],axis='columns')



dt=dc(max_depth=2)
small_model=dt.fit(X=train.iloc[:,1:],y=train.safe_loans) 

dt_l=dc(max_depth=6)
decision_tree=dt_l.fit(X=train.iloc[:,1:].as_matrix(),y=train.safe_loans)



v_safe=valid[valid.safe_loans==1]
v_risk=valid[valid.safe_loans==-1]
cl=v_safe.iloc[0:2,:].append(v_risk.iloc[0:2,:])

pr_s=small_model.predict(cl)
pr_d=decision_tree.predict(cl)

p_prob_d=decision_tree.predict_proba(X=cl)
p_prob_s=small_model.predict_proba(X=cl)

decision_tree.score(X=train_data,y=t_safe)
small_model.score(X=train_data,y=t_safe)

xt=train.iloc[:,1:]
yt=train.safe_loans
xv=valid.iloc[:,1:]
yv=valid.safe_loans

decision_tree.score(xv,yv)
small_model.score(xv,yv)


cmplx=dc(max_depth=10)
cmd=cmplx.fit(train_data,t_safe)

cmd.score(train_data,t_safe)
cmd.score(xv,yv)

p_d=decision_tree.predict(xv)
dd=pd.DataFrame({'pred':p_d,'y':yv})
dd.head()
(fn,ax)=dd[(dd.pred==-1) & (dd.y==1)].shape
(fp,ax)=dd[(dd.pred==1) & (dd.y==-1)].shape





t_safe=train['safe_loans']
del train_data['safe_loans']

dt=dc(max_depth=2)
small_model=dt.fit(X=train_data.as_matrix(),y=t_safe) 

dt_l=dc(max_depth=6)
decision_tree=dt_l.fit(X=train_data.as_matrix(),y=t_safe)
fn*10000+fp*20000




?str.strip




c_s=cl.safe_loans
del cl['safe_loans']
valid=valid[['safe_loans','short_emp',
 'emp_length_num',
 'dti',
 'last_delinq_none',
 'last_major_derog_none',
 'revol_util',
 'total_rec_late_fee',
 'grade_A',
 'grade_B',
 'grade_C',
 'grade_D',
 'grade_E',
 'grade_F',
 'grade_G',
 'sub_grade_A1',
 'sub_grade_A2',
 'sub_grade_A3',
 'sub_grade_A4',
 'sub_grade_A5',
 'sub_grade_B1',
 'sub_grade_B2',
 'sub_grade_B3',
 'sub_grade_B4',
 'sub_grade_B5',
 'sub_grade_C1',
 'sub_grade_C2',
 'sub_grade_C3',
 'sub_grade_C4',
 'sub_grade_C5',
 'sub_grade_D1',
 'sub_grade_D2',
 'sub_grade_D3',
 'sub_grade_D4',
 'sub_grade_D5',
 'sub_grade_E1',
 'sub_grade_E2',
 'sub_grade_E3',
 'sub_grade_E4',
 'sub_grade_E5',
 'sub_grade_F1',
 'sub_grade_F2',
 'sub_grade_F3',
 'sub_grade_F4',
 'sub_grade_F5',
 'sub_grade_G1',
 'sub_grade_G2',
 'sub_grade_G3',
 'sub_grade_G4',
 'sub_grade_G5',
 'home_ownership_MORTGAGE',
 'home_ownership_OTHER',
 'home_ownership_OWN',
 'home_ownership_RENT',
 'purpose_car',
 'purpose_credit_card',
 'purpose_debt_consolidation',
 'purpose_home_improvement',
 'purpose_house',
 'purpose_major_purchase',
 'purpose_medical',
 'purpose_moving',
 'purpose_other',
 'purpose_small_business',
 'purpose_vacation',
 'purpose_wedding',
 'term_36 months',
 'term_60 months']
]




