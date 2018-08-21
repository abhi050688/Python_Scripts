# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:21:26 2017

@author: Abhishek S
"""

import sframe as sf
import pandas as pd
import sklearn.linear_model as lm
import numpy as np


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float,
'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float,
'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str,
'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

train=pd.read_csv('E:/Regression/train_data.csv',sep=',',dtype=dtype_dict)
train.head()
train.dtypes
train['bedrooms_squared']=pow(train['bedrooms'],2)
train['bed_bath_rooms']=train['bedrooms']*train['bathrooms']
train['log_sqft_living']=np.log(train['sqft_living'])
train['lat_plus_long']=train['lat']+train['long']

#Means of new variables
print np.mean(train[['bedrooms_squared','bed_bath_rooms','log_sqft_living','lat_plus_long']])
    
#https://stackoverflow.com/questions/11285613/selecting-columns-in-a-pandas-dataframe

var1=['sqft_living','bedrooms','bathrooms','lat','long']
regr=lm.LinearRegression()
regr.fit(train[var1],train['price'])
coef=pd.DataFrame({'var':var1,'coef':list(regr.coef_)})

var2=var1+['bed_bath_rooms']

regr2=lm.LinearRegression()
regr2.fit(train[var2],train['price'])
coef2=pd.DataFrame({'var2':var2,'coef':regr2.coef_})


var3=var2+['bedrooms_squared','log_sqft_living','lat_plus_long']
regr3=lm.LinearRegression()
regr3.fit(train[var3],train['price'])
coef3=pd.DataFrame({'var3':var3,'coef':regr3.coef_})

print coef
print coef2
print coef3

prediction=regr2.predict(train[var2])
rss=sum(pow(train['price']-prediction,2))


def residual_sum_sqr(regression,dataset,target):
    prediction=regression.predict(dataset)
    rss=sum(pow(target-prediction,2))
    return(rss)

print residual_sum_sqr(regr,train[var1],train['price'])
print residual_sum_sqr(regr2,train[var2],train['price'])
print residual_sum_sqr(regr3,train[var3],train['price'])
    
test=pd.read_csv('E:/Regression/test_data.csv',dtype=dtype_dict)
test['bedrooms_squared']=pow(test['bedrooms'],2)
test['bed_bath_rooms']=test['bedrooms']*test['bathrooms']
test['log_sqft_living']=np.log(test['sqft_living'])
test['lat_plus_long']=test['lat']+test['long']
print residual_sum_sqr(regr,test[var1],test['price'])
print residual_sum_sqr(regr2,test[var2],test['price'])
print residual_sum_sqr(regr3,test[var3],test['price'])













‘bedrooms_squared’ = ‘bedrooms’*‘bedrooms’
‘bed_bath_rooms’ = ‘bedrooms’*‘bathrooms’
‘log_sqft_living’ = log(‘sqft_living’)
‘lat_plus_long’ = ‘lat’ + ‘long’


pd.DataFrame.
