#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:55:57 2020

@author: manpreetsi
"""



### 2nd End to End DNN Code using NumPy only ####

### This code feature additional features of DNN namely:-
#Dropout
#Normalizing/Scaling Inputs
#initializaing weights with better condition

'''
A network with following description:-
20 ------>      40 ------>           80 ------->            10 ------->             1
input_layer     Hiddent_Layer_1      Hiddent_Layer_2        Hiddent_Layer_3         Output Layer
x          w1,b1               w2,b2                 w3,b3                  w4,b4   y

hidden layer functions are RelU
Final layer function is sigmoid
'''

import numpy as np;
import math
pip install matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

learning_rate = .01

layer_nn = [20,40,80,10,1] 
nw_size = len(layer_nn)

x, y = input_data()

###Implemented Scaling the input features 
x = preprocessing.scale(x)

cost_array = []

param={}
param = param_init(layer_nn)

Z_all, A_all = {},{}
dZ_all,dW_all,db_all= {},{},{}

for epoch in range(1,1001):
                  
    Z_all, A_all = forward_prop(param,x,y,nw_size)
    A_all['A'+str(0)]=x
    
    cost = comp_cost(A_all,y,nw_size)
    cost_array.append(cost)
    dZ_all,dW_all,db_all = backward_prop(layer_nn,A_all,y,param,Z_all)
        
    param = param_update(dW_all,db_all,param,learning_rate,nw_size)

cost_array = np.array(cost_array)
cost_array = cost_array[np.isfinite(cost_array)]
xs = np.arange(1,len(cost_array)+1)

### Cost Graph #### 
plt.plot(xs,cost_array)
plt.xlabel('iterations')
plt.ylabel('Cost Function')
plt.show()



def input_data():
    x = np.random.randint(low=1, high = 10, size = 2000).reshape(20,100);
    y = np.random.randint(2, size=100).reshape(1,100)
    return x,y    
    
def param_init(layers_nn):

    np.random.seed(1)
    param={}
    size_nn = len(layer_nn)
    for i in range(1,size_nn):
        param['W' + str(i)] = np.random.random((layer_nn[i], layer_nn[i-1])) * np.sqrt(2/layer_nn[i-1]) ## initializing appropriate weights 
        param['b' + str(i)] = np.zeros((layer_nn[i],1))
        
        ### Checker to check the dimensions of weights w & bias b ###
        assert(param['W'+str(i)].shape == (layer_nn[i], layer_nn[i-1]))
        assert(param['b'+str(i)].shape == (layer_nn[i],1))

    return param

def relu(x):
    return (x>0) * x
  
def sigmoid(x):
    sig= 1/(1+ np.exp(-x))
    return sig  

def relu_deriv(x):
    print(x)
    return x>0
 
def forward_prop(param,x,y,size_nn):

    A = x
    A_prev = A   
    A_all ={}
    Z_all = {}    
    
    for i in range (1,size_nn-1):
        W = param['W'+str(i)]
        A_prev = A
        b = param['b' + str(i)]
        
        Z = np.dot(W,A_prev) + b 
        A = relu(Z)
        
        ### Implemented Dropout with 70% probability
        dropout_mask = np.random.rand(A.shape[0],A.shape[1]) < .7 ## 30% neurons will be switched off 
        A *= dropout_mask
        
        Z_all['Z' + str(i)] = Z
        A_all['A' + str(i)] = A    
     
    ## calculate output layer Z & A using Sigmoid ##   
    W = param['W'+str(size_nn - 1)]
    A_prev = A_all['A' + str(size_nn-2)]
    b = param['b' + str(size_nn - 1)]
    
    Z= np.dot(W,A_prev) + b
    A = sigmoid(Z)
    
    Z_all['Z' + str(size_nn-1)] = Z
    A_all['A'+str(size_nn-1)] = A
        
    return (Z_all, A_all)    

def comp_cost(A_all,y,size_nn):
    
    y_hat = A_all['A'+str(size_nn-1)]
    y_act = y
    m = np.size(y)
    
    cost = - np.sum((y_act*np.log(y_hat)) + (1-y_act)*np.log(1-y_hat))/m 
    np.squeeze(cost)
        
    return cost 

def backward_prop(layer_nn,A_all,y,param,Z_all):
    size_nn = len(layer_nn)
    m = np.size(y)
    dZ_all,dW_all, db_all = {},{},{}
    
    dz = A_all['A'+str(size_nn-1)] - y
    dw = (np.dot(dz,A_all['A'+str(size_nn-2)].T))/m
    db = (np.sum(dz, axis=1, keepdims=True))/m
    dZ_all['dZ' + str(size_nn-1)] = dz
    dW_all['dW' + str(size_nn-1)] = dw
    db_all['db' + str(size_nn-1)] = db
    
    for i in range(size_nn-2,0,-1):

        dz = np.dot(param['W'+str(i+1)].T,dZ_all['dZ' + str(i+1)])
        dz = dz*relu_deriv(Z_all['Z' + str(i)])
        dw = np.dot(dz,A_all['A' + str(i-1)].T)/m
        db = np.sum(dz,axis=1,keepdims=True)/m
        
        dZ_all['dZ' + str(i)] = dz
        dW_all['dW' + str(i)] = dw
        db_all['db' + str(i)] = db    
        
    return dZ_all,dW_all,db_all
    
def param_update(dW_all,db_all,param,alpha,size_nn):
    alpha = .01
    for i in range(1,size_nn):
        param['W'+str(i)] -= alpha*dW_all['dW'+str(i)]
        param['b'+str(i)] -= alpha*db_all['db'+str(i)]
    
    return param
