# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 10:35:28 2018

@author: Abhishek S
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.rnn  as rnn
from tensorflow.python.framework import ops

np.arange(0,10,.1)

class TimeSeries:
    def __init__(self,maxV,minV,num_step):
        self.maxV=maxV
        self.minV=minV
        self.num_step=num_step
        self.units=(self.maxV-self.minV)/self.num_step
        self.x=np.linspace(self.minV,self.maxV,self.num_step)
        self.y=np.sin(self.x)
    
    @staticmethod
    def ret_y(xin):
        return np.sin(xin)
    
    def next_batch(self,batch_size,steps,ret_x):
        a=np.random.randint(low=0,high=1000,size=(batch_size,1))/1000
        a=a*(self.maxV-self.minV-(steps+1)*self.units)
        batch_series=self.minV+a+np.arange(0,steps+1)*self.units
        y_batch=np.sin(batch_series)
        if ret_x:
            return y_batch[:,:-1],y_batch[:,1:],batch_series
        else:
            return y_batch[:,:-1],y_batch[:,1:]




st=TimeSeries(10,0,250)
y,y_shif,x=st.next_batch(10,30,True)
plt.plot(st.x,st.y,'-',color='b')
plt.plot(x.flatten()[:-1],y.flatten(),'*')
plt.hlines(0,st.minV,st.maxV,color='k')
plt.show()


plt.plot(x.flatten()[:-1],y.flatten(),'bo',markersize=15,alpha=.4)
plt.plot(x.flatten()[1:],y_shif.flatten(),'ko',markersize=7,alpha=1)
plt.show()


n_input=1
timestep=16
ncells=64
learning_rate=1e-3
batch_size=16
max_iter=100000
ops.reset_default_graph()
x=tf.placeholder(np.float32,[None,timestep,n_input])
y=tf.placeholder(np.float32,[None,n_input])
x_l=tf.unstack(x,timestep,axis=1)# this should provide a list of tensors of  length timestep
cell=rnn.BasicRNNCell(ncells,activation=tf.nn.relu)
output,_=rnn.static_rnn(cell,x_l,dtype='float32')
weights=tf.Variable(initial_value=tf.random_normal([ncells,n_input]))
bias=tf.Variable(initial_value=tf.zeros([n_input]))
prediction=tf.add(tf.matmul(output[-1],weights),bias)
err=tf.reduce_mean(tf.pow(tf.subtract(prediction,y),2))
opt=tf.train.AdamOptimizer(learning_rate).minimize(err)

saver=tf.train.Saver()
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for i in range(max_iter):
        num_step=np.random.randint(10,1000,1)[0]
        st=TimeSeries(50,10,num_step)
        y1,y2=st.next_batch(batch_size,timestep,False)
        y2=y2[:,-1].reshape(-1,1)
        y1=y1.reshape(batch_size,timestep,n_input)
        sess.run(opt,feed_dict={x:y1,y:y2})
        if i%10 ==0:
            loss=sess.run(err,feed_dict={x:y1,y:y2})
            print(loss)
    saver.save(sess,"E:/Tensorflow/Udemy")
            
with tf.Session() as sess:
    saver.restore(sess,"E:/Tensorflow/Udemy")
#    y3,y4,dx=st.next_batch(100,16,True)
    temp=np.array(100+np.arange(0,17)*.5)
    y3=np.sin(temp)
    pred=sess.run(prediction,feed_dict={x:y3[:-1].reshape(1,16,1)})
    print(y3[-1],pred[0,-1])

with tf.Sessios() as sess:
st=TimeSeries(200,10,500)
y3,y4,dx=st.next_batch(1,16,True)
temp_array=y3
temp_array=np.zeros([1,16])
itera=100
with tf.Session() as sess:
    saver.restore(sess,"E:/Tensorflow/Udemy")
    out=list(temp_array[0,:,])
    for i in range(itera):
        pred=sess.run(prediction,feed_dict={x:temp_array.reshape(1,16,1)})
        out.append(pred[0,0])
        temp_array=np.array(out[i+1:])
    plt.plot(out)
out


