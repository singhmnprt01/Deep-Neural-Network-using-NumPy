#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:55:57 2020

@author: manpreetsi
"""
### 3rd End to End DNN Code using NumPy only ####

### This code has additional features & Hyperparameters of DNN namely:-
# Dropout
# Normalizing/Scaling Inputs
# Initializaing weights with better condition
# Cost function vs epochs graph
# GDM
# Mini- Batch

# Train-Test Split
# Check model performance using AUC (Area under the curve)


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
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

learning_rate = .01
beta1 = .9
    
layer_nn = [20,40,80,10,1] 
nw_size = len(layer_nn)

x, y = input_data()

### Transposing because train_test_split takes columns are features and not rows !
x = x.T
y = y.T

##### Train - Test Split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.25, stratify = y)

### Transposing again to make it fit for the NN design
x_train = x_train.T
x_test  = x_test.T
y_train = y_train.T
y_test  = y_test.T


## Variable intialization
cost_array, param, Z_all, A_all, dZ_all, Vdw,Vdb  = var_init(layer_nn)

############# create mini bacthes creation begins
mini_batch_size = 32
mini_batches = []
u = x_train.shape[1]

perm = list(np.random.permutation(u))
shuffled_x = x_train[:,perm]
shuffled_y = y_train[:,perm]

num_min_batches = math.floor(u/mini_batch_size)

for k in range(0,num_min_batches):
    mini_batch_x = shuffled_x[:,k*mini_batch_size:(k+1)*mini_batch_size]
    mini_batch_y = shuffled_y[:, k*mini_batch_size:(k+1)*mini_batch_size]
    
    mini_batch=(mini_batch_x, mini_batch_y)
    mini_batches.append(mini_batch)
    
## handling the remaning datapoints
if u % mini_batch_size  != 0:
    mini_batch_x = shuffled_x[:, num_min_batches*mini_batch_size: ]
    mini_batch_y = shuffled_y[:, num_min_batches*mini_batch_size: ]
    mini_batch = (mini_batch_x, mini_batch_y)
    mini_batches.append(mini_batch)

# number of mini-batches created !
len(mini_batches)
## print 1st mini-batch x's and y
mini_batches[0][0]
mini_batches[0][1]
############# min-bacthes creation ends
  

### Run the DNN
print("Network Modeling started at ", datetime.now())
cost_array,Z_all,A_all,dZ_all,Vdw,Vdb, param,alpha = run_nn_epochs(Z_all, A_all,param,cost_array,dZ_all,Vdw,Vdb,nw_size,layer_nn,learning_rate,mini_batches)

### Run the NN for validation data (to predict and check it's AUC- Area Under the Curve)
auc_nn,Z_test_all,A_test_all = model_validation(param,x_test,y_test,nw_size)
print("DNN auc value is : ", auc_nn)

### Cost Graph Function
cost_graph(cost_array)
print("Network Modeling completed at ", datetime.now(), "minimum cost function value of ",min(cost_array), "at index value ", cost_array.index(min(cost_array)))



def run_nn_epochs(Z_all, A_all,param,cost_array,dZ_all,Vdw,Vdb,nw_size,layer_nn,learning_rate, mini_batches):
    num_batches = len(mini_batches)
    alpha = learning_rate
    
    for epoch in range(1,1001):
       cost = 0
       for num in range(0,num_batches):
           
            x_min = mini_batches[num][0] 
            y_min = mini_batches[num][1]     
            
            Z_all, A_all = forward_prop(param,x_min,y_min,nw_size)
            A_all['A'+str(0)]=x_min
            
            temp = comp_cost(A_all,y_min,nw_size)
            cost += temp
            
            dZ_all,Vdw,Vdb = backward_prop(layer_nn,A_all,y_min,param,Z_all,beta1)
            
            param = param_update(Vdw,Vdb,param,alpha,nw_size)
            
            
       cost = cost/num_batches   
       cost_array.append(cost)
       
       #alpha0 = (1/(1 + (decay_rate*epoch)))*alpha
       #alpha  = alpha0
    
       
    return (cost_array,Z_all,A_all,dZ_all,Vdw,Vdb, param,alpha)    

def model_validation(param,x_test,y_test,nw_size):
    
    Z_test_All, A_test_all = {},{}
    Z_test_all, A_test_all = forward_prop_test(param,x_test,y_test,nw_size)
    y_test_hat = A_test_all['A'+str(nw_size-1)]
    auc_nn = round(roc_auc_score(np.squeeze(y_test),np.squeeze(y_test_hat)),3)
    return auc_nn,Z_test_all,A_test_all

def cost_graph(cost_array):
    
    cost_array = np.array(cost_array)
    cost_array = cost_array[np.isfinite(cost_array)]
    xs = np.arange(1,len(cost_array)+1)
    
    ### Cost Graph #### 
    plt.plot(xs,cost_array)
    plt.xlabel('iterations')
    plt.ylabel('Cost Function')
    plt.show()
    print("################################# Cost Graph has been plotted ! ################################# ")
    
def input_data():
    
    
    x = np.random.randint(low=1, high = 10, size = 10000).reshape(20,500);
    y = np.random.randint(2, size=500).reshape(1,500)
    
    x = preprocessing.scale(x) ###Implemented Scaling the input features 
    
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

def var_init(layer_nn):
    
    cost_array = []
    
    param={}
    param = param_init(layer_nn)
        
    Z_all, A_all = {},{}
    dZ_all,Vdw, Vdb = {},{},{}
    return cost_array, param, Z_all, A_all, dZ_all, Vdw,Vdb

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

def backward_prop(layer_nn,A_all,y,param,Z_all,beta1):
    size_nn = len(layer_nn)
    m = np.size(y)
    dZ_all,dW_all, db_all = {},{},{}
    Vdw, Vdb, Sdw, Sdb = {},{},{},{}
          
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
        
    ## initializaing Gradient Descent Momemtum (GDM) Parameters
    for i in range(1,size_nn):
        Vdw["dW"+ str(i)] = np.zeros_like(dW_all["dW" + str(i)])
        Vdb["db" + str(i)] = np.zeros_like(db_all["db" + str(i)])
    
    ## updatng Gradient Momentum parameters
    for i in range(1,size_nn):
        Vdw["dW"+ str(i)] = beta1*Vdw["dW"+ str(i)] + (1-beta1)*dW_all["dW"+ str(i)]
        Vdb["db"+ str(i)] = beta1*Vdb["db"+ str(i)] + (1-beta1)*db_all["db"+ str(i)]
        
    
    return (dZ_all,Vdw,Vdb)
    
def param_update(Vdw,Vdb,param,alpha,size_nn):
    
    alpha = .01
    for i in range(1,size_nn):
        param['W'+str(i)] -= alpha*Vdw['dW'+str(i)] # using GDM parameters to updated weight
        param['b'+str(i)] -= alpha*Vdb['db'+str(i)] # using GDM parameters to updated bias
    
    return param

def forward_prop_test(param,x,y,size_nn):

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



