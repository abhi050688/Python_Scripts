# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:00:54 2018

@author: Abhishek S
"""
import os
os.chdir('S:/Anaconda/Python_Scripts')
import NN as nn
import math
import numpy as np
#X_c,Y_c=X_train,Y_tr
#hidden_layer_list=[4,4,4]
#activation_list=['relu','relu','tanh','sigmoid']
#batch_size=None
#l2_norm=0
#learning_rate=1e-3
def neural_network_with_batch(X_c,Y_c,hidden_layer_list,activation_list,learning_rate,l2_norm,keep_prob=1,max_iter=1000,batch_size=None,\
                              beta1=0.9,beta2=0.999,seed=10,iter_to_print=100,optimization='gd',epsilon=1e-8,verbose=False):
    m=X_c.shape[1]
    nx=X_c.shape[0]
    ny=Y_c.shape[0]
    if batch_size is None:
        batch_size=m
    epochs=list()
    tb=int(math.ceil(float(m)/batch_size))
    for i in xrange(tb):
        epochs.append((X_c[:,i*batch_size:(i+1)*batch_size],Y_c[:,i*batch_size:(i+1)*batch_size,]))
    epochs=tuple(epochs)
    np.random.seed(seed)
    if(type(keep_prob)==int):
        keep_prob=np.repeat(float(keep_prob),len(hidden_layer_list)+1)
    keep_prob[len(keep_prob)-1]=1
    assert(len(keep_prob)==len(hidden_layer_list)+1)
    parameter,momentum,RMS=initialize_parameter_rms(nx,ny,activation_list,hidden_layer_list)
    cost_series=list()
    for i in range(max_iter):
        for epoch in epochs:
            (X,Y)=epoch
            m=X.shape[1]
            AL,linear_cache=nn.forward_propagation_(X,hidden_layer_list,parameter,keep_prob,activation_list)
            cost=nn.compute_cost(AL,Y,parameter,l2_norm)
            if(math.isnan(cost)):
                print("Cost went nuts")
                break
#            grads=nn.backward_propagation_(Y,AL,linear_cache,keep_prob)
            grads=nn.backward_propagation_both(Y,AL,linear_cache,keep_prob)
            if(optimization=='gd'):
                parameter=nn.update_parameter(parameter,grads,learning_rate,l2_norm,m)
            elif(optimization =='gdm'):
                parameter,momentum=update_parameter_momentum(parameter,grads,momentum,learning_rate,l2_norm,m,beta1,i+1)
            elif(optimization=='adam'):
                parameter,momentum,RMS=update_parameter_adam(parameter,grads,momentum,RMS,learning_rate,l2_norm,m,beta1,beta2,i+1,epsilon)
        AL_c,_=nn.forward_propagation_(X_c,hidden_layer_list,parameter,activation_list)
        cost_t=nn.compute_cost(AL_c,Y_c,parameter,l2_norm)
        cost_series.append(cost_t)
        if(verbose and i%iter_to_print==0):
            print("Iteration: %s Cost=%.5f"%(i,cost_t))
    cache=(parameter,hidden_layer_list,activation_list)
    AL,_=nn.forward_propagation_(X_c,hidden_layer_list,parameter,activation_list)
    return AL,cache,cost_series

def initialize_parameter_rms(nx,ny,activation_list,hidden_layer_list=list()):
    layers=[nx]+hidden_layer_list+[ny]
    parameter={}
    momentum={}
    RMS={}
    factor=0.
    for i in range(1,len(layers)):
        if(activation_list[i-1]=='relu' or activation_list[i-1]=='sigmoid'):
            factor=np.sqrt(2./layers[i-1])
        else:
            factor=np.sqrt(1./layers[i-1])
        parameter['W'+str(i)]=np.random.randn(layers[i],layers[i-1])*factor
        parameter['b'+str(i)]=np.zeros([layers[i],1])
        momentum['dW'+str(i)]=np.zeros([layers[i],layers[i-1]])
        momentum['db'+str(i)]=np.zeros([layers[i],1])
    RMS=momentum.copy()
    return parameter,momentum,RMS

def update_parameter_adam(parameter,gradient,momentum,RMS,learning_rate,l2_norm,m,beta1,beta2,iteration,epsilon):
    l=len(parameter)/2
    momentum_corrected={}
    RMS_corrected={}
    for i in range(l):
        momentum['dW'+str(i+1)]=beta1*momentum['dW'+str(i+1)]+(1-beta1)*gradient['dW'+str(i+1)]
        momentum_corrected['dW'+str(i+1)]=momentum['dW'+str(i+1)]/(1 - beta1**iteration)
        momentum['db'+str(i+1)]=beta1*momentum['db'+str(i+1)]+(1-beta1)*gradient['db'+str(i+1)]
        momentum_corrected['db'+str(i+1)]=momentum['db'+str(i+1)]/(1 - beta1**iteration)
        RMS['dW'+str(i+1)]=beta2*RMS['dW'+str(i+1)]+(1-beta2)*gradient['dW'+str(i+1)]**2
        RMS_corrected['dW'+str(i+1)]=RMS['dW'+str(i+1)]/(1 - beta2**iteration)
        RMS['db'+str(i+1)]=beta2*RMS['db'+str(i+1)]+(1-beta2)*np.power(gradient['db'+str(i+1)],2)
        RMS_corrected['db'+str(i+1)]=RMS['db'+str(i+1)]/(1 - beta2**iteration)
        parameter['W'+str(i+1)]=parameter['W'+str(i+1)]*(1-learning_rate*l2_norm/m) - learning_rate*momentum_corrected['dW'+str(i+1)]/(RMS_corrected['dW'+str(i+1)]+epsilon)**0.5
        parameter['b'+str(i+1)]=parameter['b'+str(i+1)]*(1-learning_rate*l2_norm/m) - learning_rate*momentum_corrected['db'+str(i+1)]/(RMS_corrected['db'+str(i+1)]+epsilon)**0.5
    return parameter,momentum,RMS

def update_parameter_momentum(parameter,gradient,momentum,learning_rate,l2_norm,m,beta1,iteration):
    l=len(parameter)/2
    momentum_corrected={}
    for i in range(l):
        momentum['dW'+str(i+1)]=beta1*momentum['dW'+str(i+1)]+(1-beta1)*gradient['dW'+str(i+1)]
        momentum_corrected['dW'+str(i+1)]=momentum['dW'+str(i+1)]/(1 - beta1**iteration)
        momentum['db'+str(i+1)]=beta1*momentum['db'+str(i+1)]+(1-beta1)*gradient['db'+str(i+1)]
        momentum_corrected['db'+str(i+1)]=momentum['db'+str(i+1)]/(1 - beta1**iteration)
        parameter['W'+str(i+1)]=parameter['W'+str(i+1)]*(1-learning_rate*l2_norm/m) - learning_rate*momentum_corrected['dW'+str(i+1)]
        parameter['b'+str(i+1)]=parameter['b'+str(i+1)]*(1-learning_rate*l2_norm/m) - learning_rate*momentum_corrected['db'+str(i+1)]
    return parameter,momentum



#def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
#                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
#    """
#    Update parameters using Adam
#    
#    Arguments:
#    parameters -- python dictionary containing your parameters:
#                    parameters['W' + str(l)] = Wl
#                    parameters['b' + str(l)] = bl
#    grads -- python dictionary containing your gradients for each parameters:
#                    grads['dW' + str(l)] = dWl
#                    grads['db' + str(l)] = dbl
#    v -- Adam variable, moving average of the first gradient, python dictionary
#    s -- Adam variable, moving average of the squared gradient, python dictionary
#    learning_rate -- the learning rate, scalar.
#    beta1 -- Exponential decay hyperparameter for the first moment estimates 
#    beta2 -- Exponential decay hyperparameter for the second moment estimates 
#    epsilon -- hyperparameter preventing division by zero in Adam updates
#
#    Returns:
#    parameters -- python dictionary containing your updated parameters 
#    v -- Adam variable, moving average of the first gradient, python dictionary
#    s -- Adam variable, moving average of the squared gradient, python dictionary
#    """
#    
#    L = len(parameters) // 2                 # number of layers in the neural networks
#    v_corrected = {}                         # Initializing first moment estimate, python dictionary
#    s_corrected = {}                         # Initializing second moment estimate, python dictionary
#    
#    # Perform Adam update on all parameters
#    for l in range(4):
##        print l
#        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
#        ### START CODE HERE ### (approx. 2 lines)
#        v["dW" + str(l+1)] = beta1*v["dW"+str(l+1)]+(1-beta1)*grads["dW"+str(l+1)]
#        v["db" + str(l+1)] = beta1*v["db"+str(l+1)]+(1-beta1)*grads["db"+str(l+1)]
#        ### END CODE HERE ###
#
#        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
#        ### START CODE HERE ### (approx. 2 lines)
#        v_corrected["dW" + str(l+1)] = v["dW"+str(l+1)]/(1 - (beta1)**t)
#        v_corrected["db" + str(l+1)] = v["db"+str(l+1)]/(1-(beta1)**t)
#        ### END CODE HERE ###
#
#        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
#        ### START CODE HERE ### (approx. 2 lines)
#        s["dW" + str(l+1)] = beta2*s["dW"+str(l+1)]+(1-beta2)*(grads["dW"+str(l+1)])**2
#        s["db" + str(l+1)] = beta2*s["db"+str(l+1)]+(1-beta2)*(grads["db"+str(l+1)])**2
#        ### END CODE HERE ###
#
#        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
#        ### START CODE HERE ### (approx. 2 lines)
#        s_corrected["dW" + str(l+1)] = s["dW"+str(l+1)]/(1 - (beta2)**t)
#        s_corrected["db" + str(l+1)] = s["db"+str(l+1)]/(1 - (beta2)**t)
#        ### END CODE HERE ###
##        print "2nd break"
#
#        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
#        ### START CODE HERE ### (approx. 2 lines)
#        parameters["W" + str(l+1)] = parameters["W"+str(l+1)]-learning_rate*v_corrected["dW"+str(l+1)]/(s_corrected["dW"+str(l+1)]+epsilon)**0.5
##        print "3rd break"
#        parameters["b" + str(l+1)] = parameters["b"+str(l+1)]-learning_rate*v_corrected["db"+str(l+1)]/(s_corrected["db"+str(l+1)]+epsilon)**0.5
#        ### END CODE HERE ###
##        print "4th break"
#
#    return parameters, v, s
#
#parameters=parameter.copy()
