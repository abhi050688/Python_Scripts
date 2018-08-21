# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:48:18 2017

@author: Abhishek S
"""
from random import randint
import numpy as np
from sklearn.preprocessing import Normalizer
import pandas as pd
import sklearn.preprocessing as pr
from sklearn.preprocessing import StandardScaler as SS
import random

 


x=[ randint(-100,100) for i in range(10) ]
y=pd.DataFrame({'rnd':x})
Normalizer(y,axis=0)
a=pd.DataFrame(pr.normalize(y,axis=0))
a.iloc[:,0].mean()
a.iloc[:,0].sum()
np.sqrt(sum(pow(a.iloc[:,0],2)))

b=pd.DataFrame(pr.normalize(y,norm='l1',axis=0))
b.iloc[:,0].apply(lambda x:abs(x)).sum()

#l2-norm is unit euclidean distance. Consider a vector in a space. To convert it into unit distance, we use l2-norm i.e. divide it by sqrt(a^2+b^2)
#l1-norm is unit Manhattan distance. So in this case, the vector is divided by abs(a)+abs(b)
#Standardization is centering with unit variance
#Axis=0 implies a feature(Column) while Axis =1 implies a sample(Row)  
#http://scikit-learn.org/stable/modules/preprocessing.html

c=pd.DataFrame(pr.scale(y,axis=0))

#How to standardise training and test data
x=pd.DataFrame({'a':[randint(-10,10) for i in range(20)]})
y=pd.DataFrame({'b':[randint(-10,10) for i in range(20)]})

scalar=SS().fit(x)
x_scalar=scalar.transform(x)
y_scalar=scalar.transform(y)
scalar.mean_
scalar.var_

state=list(np.repeat('PA',5))
total=[randint(10000,500000) for i in range(5)]
Obama=[round(randint(0,300)/3.0,3) for i in range(5)]
Romney=[round(randint(0,300)/3.0,3) for i in range(5)]
winner=[ randint(0,1) for i in range(5) ]

election=pd.DataFrame({'state':state,'total':total,'Obama':Obama,'Romney':Romney,'winner':winner})
election['winner']= election['winner'].apply(lambda x: 'Romney' if x==1 else 'Obama')
election['country']=['Adams','Allegheny','Armstrong','Beaver','Bedford']
election.set_index('country',inplace=True)


#find nulls
##First way
sum(amazon['review'].isnull())
idxn=amazon[amazon['review'].isnull()].index
amazon.iloc[idxn,1]=''
amazon.iloc[idxn,:]

##Second way
?pd.DataFrame.fillna
?pd.Series.fillna
amazon['review'].fillna(value='',inplace=True)
sum(amazon['review'].isnull())
amazon.head()
amazon.iloc[idxn,:]


