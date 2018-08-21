# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 22:16:17 2018

@author: Abhishek S
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os

script='E:/Python'
os.chdir(script)
column=['age',
'workclass',
'fnlwgt',
'education',
'education-num',
'marital-status',
'occupation',
'relationship',
'race',
'sex',
'capital-gain',
'capital-loss',
'hours-per-week',
'native-country',
'label'
]
census=pd.read_csv('census.csv',names=column)
census.head()
census.shape
census=census.dropna(axis=0)
wkcls=census.workclass.unique()
edu=census.education.unique()
ms=census['marital-status'].unique()
rela=census.relationship.unique()
rc=census.race.unique()
sx=census.sex.unique()
census.info()

#Categorical columns
wrkclass=tf.feature_column.categorical_column_with_vocabulary_list('workclass',vocabulary_list=wkcls)
education=tf.feature_column.categorical_column_with_vocabulary_list('education',edu)
mari=tf.feature_column.categorical_column_with_vocabulary_list('marital-status',ms)
occupation=tf.feature_column.categorical_column_with_hash_bucket('occupation',hash_bucket_size=10)
relationship=tf.feature_column.categorical_column_with_vocabulary_list('relationship',rela)
race=tf.feature_column.categorical_column_with_vocabulary_list('race',rc)
sex=tf.feature_column.categorical_column_with_vocabulary_list('sex',sx)
native_country=tf.feature_column.categorical_column_with_hash_bucket('native-country',100)

#Numerical columns
wt=tf.feature_column.numeric_column('fnlwgt')
enum=tf.feature_column.numeric_column('education-num')
cpg=tf.feature_column.numeric_column('capital-gain')
cl=tf.feature_column.numeric_column('capital-loss')
hpw=tf.feature_column.numeric_column('hours-per-week')
age=tf.feature_column.numeric_column('age')
census['age'].plot('hist',bins=20)
boundry=list(set(range(20,105,5)))
boundry.sort()
age_bucket=tf.feature_column.bucketized_column(age,boundry)
base_column=[wrkclass,education,mari,occupation,relationship,race,sex]
crossed=[tf.feature_column.crossed_column(['education','occupation'],hash_bucket_size=1000),tf.feature_column.crossed_column([age_bucket,'education','occupation'],1000)]

x=census.drop('label',axis=1)
mp={' >50K':1,' <=50K':0}
y=census['label'].map(mp)
input_func=tf.estimator.inputs.pandas_input_fn(x,y,batch_size=64,num_epochs=None,shuffle=True)
model=tf.estimator.LinearClassifier(base_column+crossed)
model.train(input_func,steps=1000)

test_input=tf.estimator.inputs.pandas_input_fn(x,y,batch_size=64,num_epochs=None,shuffle=False)
test=model.evaluate(test_input,steps=500)
print(test)

 