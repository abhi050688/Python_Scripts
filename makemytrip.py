# -*- coding: utf-8 -*-
"""
Created on Tue May 22 23:09:08 2018

@author: Abhishek S
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import tensorflow as  tf
from sklearn.neural_network import MLPClassifier
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




os.chdir("E:/Analytics Vidya/MakeMyTrip/dataset")

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.head()

train.info()
train.groupby(['D','E'])['D'].count()
train.groupby(['F','G'])['F'].count()
del train['E'] #corr 1 with D
del test['E']
target='P'
i=0
train.groupby('P')[target].count()/train.shape[0]
cat=list(train.columns[train.dtypes==np.object])
def prop(train,target):
    cnt=train.groupby(target)[target].count()
    prop=round(cnt*100/train.shape[0],2)
    return pd.DataFrame({"cnt":cnt,"prop":prop})
for i in range(len(cat)):
    print(train.groupby(cat[i]).apply(lambda x:prop(x,target)))


train.describe()

mn_b=np.nanmedian(train.B)
mn_n=np.nanmedian(train.N)
fill_dict={'A':'b','D':'u','F':'NA','G':'NA','B':mn_b,'N':mn_n}
ntrain=train.fillna(fill_dict)
ntrain.loc[ntrain.F.isin(['NA','r','j']),'F']='new'
f_mean=ntrain.groupby('F')[target].mean()
f_mean=pd.DataFrame(f_mean)
f_mean.reset_index(inplace=True)
f_mean.head()
g_mean=ntrain.groupby('G')[target].mean()
g_mean=pd.DataFrame(g_mean)
g_mean.reset_index(inplace=True)

def set_data(ntrain,fill_dict,f_mean,g_mean):
    ntrain=train.fillna(fill_dict)
    ntrain.loc[ntrain.F.isin(['NA','r','j']),'F']='new'
    ntrain=ntrain.merge(f_mean,how='left',on='F')
    ntrain=ntrain.merge(g_mean,how='left',on='G')
    ntrain.rename(columns={"P":"P_G","P_x":"P","P_y":"P_F"},inplace=True)
    ntrain.drop(columns=['F','G'],axis=1,inplace=True)
    return(ntrain)

ntrain.head()
ntrain=set_data(ntrain,fill_dict,f_mean,g_mean)
ntest=set_data(test,fill_dict,f_mean,g_mean)
ntest.info()

cat=list(ntrain.columns[ntrain.dtypes==np.object])
ntrain_model=pd.get_dummies(ntrain,prefix=cat,prefix_sep='_')


x=ntrain_model.copy()
y_train=x.P
del x['P']
del x['id']
x.head()
regr=RandomForestClassifier(n_estimators=5000,criterion='gini',max_depth=2,oob_score=True,n_jobs=-1,random_state=1220,verbose=True)
regr.fit(x,y_train)
regr.oob_score_
y_predict=regr.predict(x)
sum(y_train==y_predict)/y_train.shape[0]
regr.max_features
len(regr.feature_importances_)
importance=pd.DataFrame({'features':x.columns,'score':regr.feature_importances_})
importance.sort_values('score',ascending=False,inplace=True)
from sklearn.preprocessing import normalize
importance['nrm']=normalize(importance[['score']],axis=0)
importance.iloc[0:12,0].values
importance.iloc[0:10,0]
y_train.head()
i=0
def cross_validation(x,y_train,bands=10):
    perm=np.random.permutation(len(x))
    x=x.iloc[perm,:]
    y_train=y_train[perm]
    y_train.reset_index(inplace=True,drop=True)
    x.reset_index(inplace=True,drop=True)
    size=int(x.shape[0]/bands)
    size_array=np.arange(0,x.shape[0],size)
    size_array[bands]=x.shape[0]
    err=0
    for i in range(bands):
        validation_x=x.iloc[size_array[i]:size_array[i+1],:]
        validation_y=y_train[size_array[i]:size_array[i+1]]
        train_x=pd.concat([x.iloc[:size_array[i],:],x.iloc[size_array[i+1]:,:]],axis=0)
        train_y=pd.concat([y_train[:size_array[i]],y_train[size_array[i+1]:]],axis=0)
        regr=RandomForestClassifier(n_estimators=5000,criterion='gini',max_depth=2,oob_score=True,n_jobs=-1,random_state=1220,verbose=True)
        regr.fit(train_x,train_y)
        v_predict=regr.predict(validation_x)
        err+=sum(v_predict!=validation_y)
    accuracy=err/x.shape[0]
    return(accuracy)

acc=cross_validation(x,y_train,bands=10)

nn=MLPClassifier(hidden_layer_sizes=(5,5,4),max_iter=300000,random_state=1002,shuffle=False)
nn.fit(x,y_train)
y_predict=nn.predict(x)
sum(y_predict==y_train)/y_train.shape[0]
x.head()



ntrain_model.head()
x_train,x_test,y_train,y_test=train_test_split(ntrain_model.drop(columns=['P','id'],axis=1),ntrain_model.P,test_size=.25)

x=(x_train.T).values
x.shape
y=(y_train.values).reshape(1,-1)



#x,y=make_moons(n_samples=300,noise=0.8,random_state=102)
#x.shape
#y.shape
#scl=StandardScaler()
#x=scl.fit_transform(x)
#x=x.T
#y=y.reshape(1,-1)
#
def training_model(x,y,x_t,y_t,max_iter=5000):
    nx=x.shape[0]
    ny=y.shape[0]
    layer=[nx]+[5,5]+[ny]
    ops.reset_default_graph()
    X=tf.placeholder(np.float32,shape=(layer[0],None),name="X")
    Y=tf.placeholder(np.float32,shape=(layer[len(layer)-1],None),name="Y")
    W1=tf.get_variable("W1",shape=(layer[1],layer[0]),initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1=tf.get_variable("b1",shape=(layer[1],1),initializer=tf.zeros_initializer())
    W2=tf.get_variable("W2",shape=(layer[2],layer[1]),initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2=tf.get_variable("b2",shape=(layer[2],1),initializer=tf.zeros_initializer())
    W3=tf.get_variable("W3",shape=(layer[3],layer[2]),initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3=tf.get_variable("b3",shape=(layer[3],1),initializer=tf.zeros_initializer())
    parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}
    Z1=tf.add(tf.matmul(W1,X),b1)
    A1=tf.nn.relu(Z1)
    Z2=tf.add(tf.matmul(W2,A1),b2)
    A2=tf.nn.relu(Z2)
    Z3=tf.add(tf.matmul(W3,A2),b3)
    logits=tf.transpose(Z3)
    labels=tf.transpose(Y)
    cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    init=tf.global_variables_initializer()
    costs=list()
    sess=tf.Session()
    sess.run(init)
    for i in range(max_iter):
        _,cost_iter=sess.run([optimizer,cost],feed_dict={X:x,Y:y})
#        if(i%20==0):
#            costs.append(cost_iter)
    A3=tf.nn.sigmoid(Z3)
    A3=(A3>=0.5)
    correct=tf.equal(tf.cast(A3,"float32"),Y)
    net_correct=correct.eval({X:x_t,Y:y_t},session=sess).sum()
    return  net_correct

def cross_validation_tf()





A3_mat=sess.run(A3,feed_dict={X:x,Y:y})
A3_mat=A3_mat>=0.5


correct=tf.equal(tf.argmax(Z3),tf.argmax(Y))
accuracy=tf.reduce_mean(tf.cast(correct,"float"))
print("Accuracy: ",accuracy.eval({X:x,Y:y},session=sess))

x_train.columns

x_t=(x_test.T).values
y_t=(y_test.values).reshape(1,-1)
x_t.shape

A3=tf.nn.sigmoid(Z3)
A3.eval({X:x_t,Y:y_t},session=sess).shape


print("Test Accuracy: ",accuracy.eval({X:x_t,Y:y_t},session=sess))

parameter=sess.run(parameters)

prediction=tf.nn.sigmoid(Z3)

prediction=sess.run(prediction)
sess.run()





sess.run(cost,feed_dict={X:x,Y:y})

a3=tf.nn.sigmoid(Z3)
a3=sess.run(a3,feed_dict={X:x,Y:y})
a3.shape



par=sess.run(parameters)
par['W1']
sess.run(Z3)
tf.arg_max(Z3)
Y
corr=tf.equal(tf.argmax(Z3),tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(corr, "float"))

        print ("Train Accuracy:", accuracy.eval({X:x, Y:y}))


            
plt.plot(np.squeeze(costs))
plt.ylabel("cost")
plt.xlabel("iterations (per tens)")
plt.title("learning rate = "+str(1e-4))
plt.show()
        
    
W=tf.get_variable(name='W',dtype=np.float32,shape=(1,1))
y=tf.constant(5.0)
cost=tf.pow(tf.subtract(y,W),2)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
sess.run(optimizer)
sess.run(W)
print(W)

for i  in range(1000):
    sess.run(optimizer)

sess.run(W)



















