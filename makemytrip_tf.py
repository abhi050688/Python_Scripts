# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 22:55:16 2018

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
ntrain_model.head()

x_train,x_test,y_train,y_test=train_test_split(ntrain_model.drop(columns=['P','id'],axis=1),ntrain_model.P,test_size=.25)




#x,y=make_moons(n_samples=300,noise=0.8,random_state=102)
#x.shape
#y.shape
#scl=StandardScaler()
#x=scl.fit_transform(x)
#x=x.T
#y=y.reshape(1,-1)
#
def training_model(x,y,x_t,y_t,hidden_layer=[5,5,5],learning_rate=1e-4,max_iter=5000,l2_norm=1e-2):
    nx=x.shape[0]
    ny=y.shape[0]
    layer=[nx]+hidden_layer+[ny]
    ops.reset_default_graph()
    X=tf.placeholder(np.float32,shape=(layer[0],None),name="X")
    Y=tf.placeholder(np.float32,shape=(layer[len(layer)-1],None),name="Y")
    W1=tf.get_variable("W1",shape=(layer[1],layer[0]),initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1=tf.get_variable("b1",shape=(layer[1],1),initializer=tf.zeros_initializer())
    W2=tf.get_variable("W2",shape=(layer[2],layer[1]),initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2=tf.get_variable("b2",shape=(layer[2],1),initializer=tf.zeros_initializer())
    W3=tf.get_variable("W3",shape=(layer[3],layer[2]),initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3=tf.get_variable("b3",shape=(layer[3],1),initializer=tf.zeros_initializer())
    W4=tf.get_variable("W4",shape=(layer[4],layer[3]),initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4=tf.get_variable("b4",shape=(layer[4],1),initializer=tf.zeros_initializer())
    parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4}
    Z1=tf.add(tf.matmul(W1,X),b1)
    A1=tf.nn.relu(Z1)
    Z2=tf.add(tf.matmul(W2,A1),b2)
    A2=tf.nn.relu(Z2)
    Z3=tf.add(tf.matmul(W3,A2),b3)
    A3=tf.nn.relu(Z3)
    Z4=tf.add(tf.matmul(W4,A3),b4)
    logits=tf.transpose(Z4)
    labels=tf.transpose(Y)
    cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
    regularizers=tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)
    cost=tf.reduce_mean(cost+l2_norm*regularizers)
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init=tf.global_variables_initializer()
    costs=list()
    sess=tf.Session()
    sess.run(init)
    for i in range(max_iter):
        _,cost_iter=sess.run([optimizer,cost],feed_dict={X:x,Y:y})
        if(i%20==0):
            costs.append(cost_iter)
    A4=tf.nn.sigmoid(Z4)
    A4=(A4>=0.5)
    correct=tf.equal(tf.cast(A4,"float32"),Y)
    net_correct=correct.eval({X:x_t,Y:y_t},session=sess).sum()
    accuracy=tf.reduce_mean(tf.cast(correct,"float"))
    print("Training Accuracy  =: ",accuracy.eval({X:x,Y:y},session=sess))
    print("Testing Accuracy =: ",accuracy.eval({X:x_t,Y:y_t},session=sess))
    net_cost=cost.eval({X:x,Y:y},session=sess)
    sess.close()
    return net_correct,net_cost

x_train.columns
scl=StandardScaler()
x_train=scl.fit_transform(x_train)
x_test=scl.transform(x_test)
x=(x_train.T)
x.shape
y=(y_train.values).reshape(1,-1)


x_t=(x_test.T)
y_t=(y_test.values).reshape(1,-1)
x_t.shape


sess,net_correct,costs=training_model(x,y,x_t,y_t,1e-4,100000)
sess.close()
net_correct
plt.plot(np.squeeze(costs))
plt.ylabel("cost")
plt.xlabel("iterations (per tens)")
plt.title("learning rate = "+str(5e-3))
plt.show()


i=0
def cross_validation_tf(ntrain_model,hidden_layer,learning_rate=1e-4,max_iter=100000,l2_norm=1e-2):
    ln=ntrain_model.shape[0]
    k=10
    ln_int=int(ln/k)
    a=list(range(0,ln,ln_int))
    a[len(a)-1]=ln
    corr=0
    for i in range(len(a)-1):
        testing=ntrain_model.iloc[a[i]:a[i+1],:]
        training=ntrain_model.loc[~ntrain_model.index.isin(testing.index),:]
        print("testing index: ",testing.shape)
        print("training_index: ",training.shape)
        scl=StandardScaler()
        x=(scl.fit_transform(training.drop(columns=['id','P']))).T
        y=training.P.values.reshape(1,-1)
        x_t=scl.transform((testing.drop(columns=['id','P'],axis=1).values)).T
        y_t=testing.P.values.reshape(1,-1)
        print("=============epoch %s==================="%(i))
        net_correct,net_cost=training_model(x,y,x_t,y_t,hidden_layer,learning_rate,max_iter,l2_norm)
        corr+=net_correct
    print("Net Accuracy: %.5f and total training cost: %.5f"%(corr/ln,net_cost))
    return corr
imp=['I_f', 'I_t', 'K', 'J_f', 'J_t', 'O', 'H', 'P_F', 'N', 'C', 'P_G',
       'B','P','id']
imp_model=ntrain_model[imp]
nc=cross_validation_tf(imp_model,hidden_layer=[10,10,5],learning_rate=1e-4,max_iter=80000,l2_norm=1e-2)
477/552
nc=0 
scl=StandardScaler()
x=scl.fit_transform(imp_model.drop(columns=['P','id'],axis=1).values).T
x.shape
y=imp_model.P.values.reshape(1,-1)

corr,ncost=training_model(x,y,x,y,hidden_layer=[10,10,5],learning_rate=1e-4,max_iter=80000,l2_norm=1e-2) 

from sklearn.ensemble import RandomForestClassifier
xr=x.T
yr=y.flatten()
yr.shape
xr.shape


regr=RandomForestClassifier(n_estimators=5000,criterion='gini',max_depth=2,oob_score=True,n_jobs=-1,random_state=1220,verbose=True)
regr.fit(xr,yr)
y_pre=regr.predict_proba(xr)
y_pre=y_pre[:,1]

np.sum(-yr*np.log(y_pre)-(1-yr)*(np.log(1-y_pre)))/yr.shape[0]


correct=tf.equal(tf.argmax(Z3),tf.argmax(Y))
accuracy=tf.reduce_mean(tf.cast(correct,"float"))
print("Accuracy: ",accuracy.eval({X:x,Y:y},session=sess))


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

for i  in range(10):
    print(i)

a=[-1,-2,4,5]
a>0
dir(a)
A=[-1,-2,-3,1,2,4]
a=set(range(0,100001))
b=a.intersection(A)

c=set(range(1,max(b)+1))
len(c-b)
f=c-b
f[0]
1 in b


def solution(A):
    # write your code in Python 3.6
    A=set(A)
    if max(A)<0:
        return 1
    b=set(range(1,max(A)+1))
    c=b.intersection(A)
    d=b-c
    if len(d)>0:
        print(list(d)[0])
    else:
        print(max(A)+1)


solution()
b=set(range(1,7))
A=[4,5,6,2]


