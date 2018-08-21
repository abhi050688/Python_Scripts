# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:17:40 2017

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sframe as sf

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float,
'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float,
'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str,
'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train=sf.SFrame.read_csv('E:/Regression/train_data.csv',column_type_hints=dtype_dict)
test=sf.SFrame.read_csv('E:/Regression/test_data.csv',column_type_hints=dtype_dict)

features=['bedrooms','bathrooms']
a=train[features].to_numpy()
a[1,:]



def get_numpy_data(data_sframe,features,output):
    data_sframe['constant']=1
    new_data_frame=data_sframe[['constant']+features].to_numpy()
    output_d=data_sframe[[output]].to_numpy()
    return(new_data_frame,output_d)

def predict_outcome(feature_matrix,weights):
    prediction=np.dot(feature_matrix,weights)
    return(prediction)

def feature_derivative(feature_vector,errors):
    return(np.dot(np.transpose(feature_vector),errors))

def regression_gradient_descent(feature_matrix,tolerance,initial_weights,step_size,output):
    errors=output[:,0]-predict_outcome(feature_matrix,initial_weights)
    gradient=feature_derivative(feature_matrix,errors)
    gradient_mag=pow(sum(pow(gradient,2)),.5)
    print "gradient_mag", gradient_mag
    weights=initial_weights[:]
    k=1
    while(gradient_mag>tolerance):
        for i in range(len(weights)):
            weights[i]=weights[i] + 2*step_size*gradient[i]
        errors=output[:,0]-predict_outcome(feature_matrix,weights)
        rss=sum(pow(errors,2))
        print k,' <- KResidual Sum of Square ', rss
        gradient=feature_derivative(feature_matrix,errors)
        gradient_mag=pow(sum(pow(gradient,2)),.5)
        k=k+1
        
    return(weights)

simple_feature=['sqft_living']
output='price'
initial_weights=np.array([-47000.,1.])
step_size=7e-12
tolerance=2.5e7
            
feature_mat,output_mat=get_numpy_data(train,simple_feature,output)
prediction=predict_outcome(feature_mat,initial_weights)
error=np.array(output_mat[:,0])-predict_outcome(feature_mat,weights)
gradient=feature_derivative(feature_mat,error)
gradient[0]
weights=regression_gradient_descent(feature_mat,tolerance,initial_weights.copy(),step_size,output_mat)
test_simple_feature_matrix,out_mat=get_numpy_data(test,simple_feature,output)

test_prediction=predict_outcome(test_simple_feature_matrix,weights)
test_rss=sum(pow(out_mat[:,0]-test_prediction,2))


model_feature=['sqft_living','sqft_living15']
output='price'
initial_weights=np.array([-100000.,1.,1.])
step_size=4e-12
tolerance=1e9

test_m_matrix,out_m_mat=get_numpy_data(test,model_feature,output)
m_weights=regression_gradient_descent(test_m_matrix,tolerance,initial_weights.copy(),step_size,out_m_mat)

test_m_prediction=predict_outcome(test_m_matrix,m_weights)
out_m_mat
test_m_mat=sum(pow(out_m_mat[:,0]-test_m_prediction,2))
print test_m_mat
print test_rss








