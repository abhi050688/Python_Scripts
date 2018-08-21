# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 23:50:36 2017

@author: Abhishek S
"""

import sframe as sf
import matplotlib.pyplot as plt


data=sf.SFrame.read_csv('E:/Regression/kc_house_data.csv')
data.head()
train_data,test_data=data.random_split(.8,seed=0)

input_feature=train_data['sqft_above']
output=train_data['price']
r,c=train_data.shape
w1=float()
w0=float()
Sigma_xy=sum(input_feature*output)
Sigma_x=input_feature.sum()
Sigma_y=output.sum()
Sigma_xsquare=sum(pow(input_feature,2))
w1=(Sigma_xy - (Sigma_x * Sigma_y)/r)/(Sigma_xsquare - (pow(Sigma_x,2))/r)
w0=output.mean() - w1*input_feature.mean()
print w1, w0


def simple_linear_regression(input_feature,output):
    w1=float()
    w0=float()
    r,=output.shape
    Sigma_xy=sum(input_feature*output)
    Sigma_x=sum(input_feature)
    Sigma_y=sum(output)
    Sigma_xsquare=sum(pow(input_feature,2))
    w1=(Sigma_xy - Sigma_x*Sigma_y/r )/(Sigma_xsquare - pow(Sigma_x,2)/r)
    w0=output.mean() - w1*input_feature.mean()
    return(w0,w1)

W0,W1=simple_linear_regression(train_data['sqft_living'],train_data['price'])
print W0,W1
print simple_linear_regression(train_data['sqft_above'],train_data['price'])

def get_regression_prediction(input_feature,slope,intercept):
    output=intercept+input_feature*slope
    return(output)

def get_residual_sum_of_square(input_feature,output,slope,intercept):
    predictions=get_regression_prediction(input_feature,slope,intercept)
    rss=sum(pow((predictions-output),2))
    return(rss)
def inverse_regression_prediction(output,slope,intercept):
    input_feature=(output-intercept)/slope
    return(input_feature)
b0,b1=simple_linear_regression(train_data['bedrooms'],train_data['price'])

rss_s=get_residual_sum_of_square(test_data['sqft_living'],test_data['price'],W1,W0)
rss_b=get_residual_sum_of_square(test_data['bedrooms'],test_data['price'],b1,b0)

print rss_s,rss_b

train_data.export_csv('E:/Regression/train_data.csv')
test_data.export_csv('E:/Regression/test_data.csv')





















