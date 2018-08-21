# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:37:56 2017

@author: Abhishek S
"""

import sframe as sf
import numpy as np
import sframe.aggregate as agg
import sklearn
import sklearn.ensemble
from sklearn.ensemble import GradientBoostingClassifier


 

loans=sf.SFrame('E:/Machine learning Classification/Week 5/lending-club-data.gl')
loans
loans.print_rows(5,68)
dt={0:1,1:-1}
loans['safe_loans']=loans['bad_loans'].apply(lambda x:1 if x==0 else -1)
del loans['bad_loans']
loans
target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]
loans=loans[[target]+features]
#Separate out rows with missing values
loans,loans_w_miss=loans.dropna_split()
loans.shape
loans_w_miss.shape
safe_loans=loans[loans['safe_loans']==1]
risky_loans=loans[loans['safe_loans']==-1]
safe_loans.shape
risky_loans.shape
p=risky_loans[target].size()/float(safe_loans[target].size())
safe_loans_sample=safe_loans.sample(p,seed=1)
safe_loans_sample.shape
safe_loans.groupby(target,{'cnt':agg.COUNT})
loans_data=risky_loans.append(safe_loans_sample)
loans_data.groupby(target,{'cnt':agg.COUNT})

cat=list()
for k,v in zip(loans_data.column_names(),loans_data.column_types()):
    print v
    if v==str:
        cat.append(k)

for feature in cat:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

loans_data.column_names()
train_data,validation_data=loans_data.random_split(.8,seed=1)

train_x=train_data.copy()
train_x.remove_column(target)
train_y=train_data[target]
grd=GradientBoostingClassifier(n_estimators=5,max_depth=6)
grd.fit(X=train_x.to_numpy(),y=train_y.to_numpy())


validation_safe=validation_data[validation_data[target]==1]
validation_risk=validation_data[validation_data[target]==-1]
sample_validation_data=validation_safe[0:2].append(validation_risk[0:2])
sample_safe=sample_validation_data[target]
sample_validation_data.remove_column(target)
grd.predict(sample_validation_data.to_numpy())
grd.predict_proba(sample_validation_data.to_numpy())[:,1]

valid_x=validation_data.copy()
valid_x.remove_column(target)
valid_y=validation_data[target]
grd.score(valid_x.to_numpy(),valid_y.to_numpy())
validation_data['prediction']=grd.predict(valid_x.to_numpy())
fp=len(validation_data[(validation_data[target]==-1) & (validation_data['prediction']==1)])
fn=len(validation_data[(validation_data[target]==1) & (validation_data['prediction']==-1)])
tc=fp*20000+fn*10000
validation_data['probability']=grd.predict_proba(valid_x.to_numpy())[:,1]
bottom=validation_data.sort('probability')[0:5]
top=validation_data.sort('probability',ascending=False)[0:5]
top.print_rows(5,47)
bottom['grade']
model_10=GradientBoostingClassifier(n_estimators=10,max_depth=6)
model_50=GradientBoostingClassifier(n_estimators=50,max_depth=6)
model_100=GradientBoostingClassifier(n_estimators=100,max_depth=6)
model_200=GradientBoostingClassifier(n_estimators=200,max_depth=6)
model_500=GradientBoostingClassifier(n_estimators=500,max_depth=6)
train_x=train_x.to_numpy()
train_y=train_y.to_numpy()
model_10.fit(train_x,train_y)
model_50.fit(train_x,train_y)
model_100.fit(train_x,train_y)
model_200.fit(train_x,train_y)
model_500.fit(train_x,train_y)
valid_x=valid_x.to_numpy()
valid_y=valid_y.to_numpy()
print model_10.score(valid_x,valid_y)
print model_50.score(valid_x,valid_y)
print model_100.score(valid_x,valid_y)
print model_200.score(valid_x,valid_y)
print model_500.score(valid_x,valid_y)


import matplotlib.pyplot as plt
%matplotlib inline
def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

train_errors=[1-model_10.score(train_x,train_y),1-model_50.score(train_x,train_y),1-model_100.score(train_x,train_y),1-model_200.score(train_x,train_y),1-model_500.score(train_x,train_y)]
valid_errors=[1-model_10.score(valid_x,valid_y),1-model_50.score(valid_x,valid_y),1-model_100.score(valid_x,valid_y),1-model_200.score(valid_x,valid_y),1-model_500.score(valid_x,valid_y)]

    plt.plot([10, 50, 100, 200, 500], train_errors, linewidth=4.0, label='Training error')
    plt.plot([10, 50, 100, 200, 500], valid_errors, linewidth=4.0, label='Validation error')
    
    make_figure(dim=(10,5), title='Error vs number of trees',
                xlabel='Number of trees',
                ylabel='Classification error',
                legend='best')


#Assignment 2


















