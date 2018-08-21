# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:05:45 2017

@author: Abhishek S
"""

import sframe as sf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float,
'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float,
'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str,
'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train=pd.read_csv('E:/Regression/train_data.csv',dtype =dtype_dict)
test=pd.read_csv('E:/REgression/test_data.csv',dtype=dtype_dict)
sales=pd.read_csv('E:/Regression/kc_house_data.csv',dtype=dtype_dict)

ft=pd.DataFrame(sales['sqft_living'])
help(ft.iloc)
ft.iloc[:,0].head()
ft['sqft_living_2']=pow(ft['sqft_living'],2)
help(ft.add)
ft.columns[0]='sqft_2'
ft.rename(columns={'sqft_living':'sqft_2'},inplace=True)

def polynomial_dataframe(feature,degree):
    ft=pd.DataFrame(feature)
    nm=ft.columns[0]
    ft.rename(columns={nm:(nm+'_1')},inplace=True)
    for i in range(2,degree+1):
        ft[nm+'_'+str(i)]=pow(ft[nm+'_1'],i)
    return(ft)
new_feat=polynomial_dataframe(sales['sqft_living'],15)
new_feat['price']=sales['price']

regr=lm.LinearRegression()
regr.fit(X=new_feat[['sqft_living_1']],y=sales['price'])
#pass 2-D array instead of 1-D error resoluved using [['sqft_living']] instead of ['sqft_living']
output=regr.predict(new_feat[['sqft_living_1']])

plt.plot(new_feat['sqft_living_1'],new_feat['price'],'.',
         new_feat['sqft_living_1'],output,'-')

new_feat=polynomial_dataframe(sales['sqft_living'],15)
new_feat['price']=sales['price']
var=list(new_feat.columns)
var.remove('price')
new_feat=new_feat.sort_values('sqft_living_1')
regr_15=lm.LinearRegression()
regr_15.fit(new_feat[var],new_feat['price'])
out_15=regr_15.predict(new_feat[var])
plt.plot(new_feat['sqft_living_1'],new_feat['price'],'.',
         new_feat['sqft_living_1'],regr_15.predict(new_feat[var]),'-')
regr_15.coef_

def multi_training_data(dataframe,feature,degree):
    dataframe=dataframe.sort_values(feature)
    new_feat=polynomial_dataframe(dataframe[feature],degree)
    new_feat['price']=dataframe['price']
    var=list(new_feat.columns)
    var.remove('price')
    regr=lm.LinearRegression()
    regr.fit(new_feat[var],new_feat['price'])
    plt.plot(new_feat[var[0]],new_feat['price'],'.',
             new_feat[var[0]],regr.predict(new_feat[var]),'-')
    return(regr,var)

multi_training_data(sales,'sqft_living',15)

train_1=sales.sample(frac=.5,replace=False,random_state=0)
train_1.index
test_1=sales.drop(train_1.index)
train_2=train_1.sample(frac=.5,replace=False,random_state=0)
train_3=train_1.drop(train_2.index)
test_2=test_1.sample(frac=.5,replace=False,random_state=0)
test_3=test_1.drop(test_2.index)


regr2,var2=multi_training_data(train_2,'sqft_living',15)
regr3,var3=multi_training_data(train_3,'sqft_living',15)
regr4,var4=multi_training_data(test_2,'sqft_living',15)
regr5,var5=multi_training_data(test_3,'sqft_living',15)

regr2.coef_
regr3.coef_
regr4.coef_
regr5.coef_

sales[[feature]]=sales[feature].apply(lambda x:float(x))
sales[["price"]]=sales["price"].apply(lambda x:float(x))

train_valid=sales.sample(frac=0.9,replace=False,random_state=0)
test=sales.drop(train_valid.index)
train=train_valid.sample(frac=0.5,replace=False,random_state=0)
valid=train_valid.drop(train.index)

train.to_csv('E:/Regression/train.csv')
valid.to_csv('E:/Regression/valid.csv')
test.to_csv('E:/Regression/test.csv')
a=polynomial_dataframe(sales['sqft_living'],1)

def compute_degree(dataframe,regr,var):
    out=regr.predict(dataframe[var])
    rss=np.sqrt(sum(pow(dataframe['price']-out,2)))
    return(rss)
degree=15
feature='sqft_living'
train_feat=polynomial_dataframe(train[feature],15)
train_feat['price']=train['price']
print compute_degree(train_feat,regr2,var2)

train=train.sort_values(feature)
valid=valid.sort_values(feature)
test=test.sort_values(feature)


data=valid
for i in range(1,degree+1):
    train_feat=polynomial_dataframe(data[feature],i)
    train_feat['price']=data['price']
    regr_l,var_l=multi_training_data(train,feature,i)
    rss=compute_degree(train_feat,regr_l,var_l)
    print 'Error at dgree -> ',i ,' = ',rss


train.shape
train_feat=polynomial_dataframe(train['sqft_living'],3)
regress=lm.LinearRegression()
regress.fit(train_feat,train['price'])
out=regress.predict(train_feat)
np.sqrt(sum(pow(out-train['price'],2)))


