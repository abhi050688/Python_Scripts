# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 23:20:42 2018

@author: Abhishek S
"""



W=np.concatenate([grads['dW1'].reshape(-1,1),grads['dW2'].reshape(-1,1),grads['db1'].reshape(-1,1),grads['db2'].reshape(-1,1)])
W[0]
epsilon=1e-7
a=['W1','W2','b1','b2']
dW=list()
for i in a:
    for j in range(p[i].shape[0]):
        for k in range(p[i].shape[1]):
            p=parameter.copy()
            print p[i][j,k] 
            p[i][j,k]=p[i][j,k]+epsilon
            AL,_=forward_propagation_(X,Y,hidden_layer_list,p,activation_list)
            j_plus=compute_cost(AL,Y)
            p[i][j,k]=p[i][j,k]-epsilon
            AL,_=forward_propagation_(X,Y,hidden_layer_list,p,activation_list)
            j_minus=compute_cost(AL,Y)
            dW.append((j_plus-j_minus)/2*epsilon)
dW=np.array(dW).reshape(-1,1)

(np.sum((W-dW)**2))**0.5/(np.sum(W**2)+np.sum(dW**2))**0.5

            





clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1) 
clf.fit(X.T,Y[0])
clf.predict_proba(X.T)[0:10]
#del titanic['name']
clf.loss_



dataset=make_moons(noise=0.8,random_state=1)
X,Y=dataset
scalar=StandardScaler()
X=scalar.fit_transform(X)
X.shape
Y.shape
hidden_layer_list=[5,5,4]
activation_list=['relu','relu','tanh','sigmoid']
X=X.T
Y=Y.reshape(1,-1)
output,cache,costs=neural_network(X,Y,hidden_layer_list,activation_list,learning_rate=5e-2,max_iter=100000,verbose=True)
output,cache,costs=neural_network(X,Y,hidden_layer_list,activation_list,learning_rate=1e-1,max_iter=300000,l2_norm=1e-1,verbose=True)
probs,_=nn_predict(cache,X)
plt.plot(costs)
np.sum((output>=0.5) == Y)/float(Y.shape[1])
X.shape
np.argwhere(output==1.)
X[:,98]


A.shape

A.shape
Y.shape

decision_boundary(X,Y,cache)

np.append
plt.plot(c)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()




X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
scalar=StandardScaler()
scalar.fit(X_train)
X_train=scalar.transform(X_train)
X_test=scalar.transform(X_test)

model=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(3,3),random_state=1)
model.fit(X_train,Y_train)
accuracy=np.sum(model.predict(X_train)==Y_train)/float(len(Y_train))
accuracy_test=np.sum(model.predict(X_test)==Y_test)/float(len(Y_test))

#factor=.01
#learning_rate=1e-4
X=X_train.T
Y=Y_train.reshape(1,-1)
plt.plot(c)

AL,_=forward_propagation_(X_test.T,hidden_layer_list,p,activation_list)
np.sum((o>=0.5)==Y)/100.
np.sum((AL>=0.5)==Y_test.reshape(1,-1))/20.
AL.shape
plt.scatter(X[:,0],X[:,1],c=Y)
plt.xlabel("X1 values")
plt.ylabel("X2 values")
plt.show()
x_min,y_min





len(linear_cache)
W,b,A_prev,Z,activation=linear_cache[2]
relu(Z)
np.dot(W,A_prev)+b
np.sum(grads['dZ2'],axis=1,keepdims=True)
dz3=np.dot(W.T,grads['dZ3'])
