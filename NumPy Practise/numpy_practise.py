#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:04:29 2020

@author: manpreetsi
"""
## NumPy Practise ##


import numpy as np
pip install sklearn

print(np.__version__)
# 1.18.1


# create a numpy 1D array  from 10 to 15
myarray = np.arange(10,16)
myarray


#Create a 2*2 numpy array of all True’s
myarray = np.full((2,2),True, dtype=bool)


#Extract all odd numbers from arr
myarray = np.arange(1,21)
myarray_odd = myarray[myarray % 2 ==1 ]
myarray_odd


#Replace all odd numbers in arr with -1
myarray = np.arange(1,21)
myarray[myarray % 2 ==1 ] = -1 
myarray


#Replace all odd numbers in arr with -1 without changing arr
myarray = np.arange(1,21)
outmyarray_new = np.where(myarray%2==1,-1,myarray)
outmyarray_new
myarray


# Convert a 1D array to a 2D array with 4 rows
myarray = np.arange(1,21)
myarray_2d = myarray.reshape(4,-1) # 2 here is number of rows and -1 let the function decide the columns
myarray_2d


##Stack arrays a and b vertically
myarray1 = np.arange(10).reshape(2,-1)
myarray2 = np.repeat(1,10).reshape(2,-1)

#mthod 1
myarray_stacked = np.concatenate([myarray1,myarray2], axis=0)

#method2

myarray_stacked = np.vstack([myarray1,myarray2])


##Stack arrays a and b horizontally
#method 1
myarray_stacked = np.concatenate([myarray1,myarray2], axis =1 )

#method 2
myarray_stacked = np.hstack([myarray1,myarray2])


#How to get the common items between two python numpy arrays?
a = np.array([1,2,3,4,5,6,7,8,9])
b = np.array([5,6,7,8,5,10,11,12,9])
np.intersect1d(a,b)


# How to get the positions where elements of two arrays match?
np.where([a==b])



#Get all items between 5 and 10 from a
a = np.array([2, 6, 1, 9, 10, 3, 27])
#method 1
output = np.where((a<=10)&(a>=5))
a[output]

#method 2
a[(a<=10)&(a>=5)]


# swap two columns/rows in a 2d array
a = np.arange(9).reshape(3,3) # by default the order of column is 0,1,2 and rows are 0,1,2

a[:,[1,0,2]] # swap 0 with 1 at column level

a[[1,0,2],:] # swap 0 with 1 at row level



# reverse a rows/columns 2d array 
a= np.arange(9).reshape(3,3)
a[::-1]  # rows 
a[:,::-1]  # columns



# reverse a 1d array  -- similar to row reverse
a = np.arange(1,15)
a[::-1]


# Write a NumPy program to create a null vector of size 10 and update sixth value to 11
a = np.zeros(9).reshape(3,3)
a = np.zeros(9)
a[5]= 11
a


################# Linear Algebra with NumPy #######################

# multiplication of 2 matrixes:-
a = np.array([1,2,3,4]).reshape(2,2)
b = np.array([5,6,7,8]).reshape(2,2)
np.dot(a,b)
np.dot(a,b.T)


# Compute the outer product of two given vectors
a = np.array([1,2,3,4]).reshape(2,2)
b = np.array([5,6,7,8]).reshape(2,2)

a*b # normal product
np.outer(a,b) # outer product


#Write a NumPy program to find a matrix or vector norm.
a = np.array([1,2,3,4]).reshape(2,2)
np.shape(a)
norm = np.linalg.norm(a, axis=1, ord=2, keepdims=True)
b = a/norm

## or ##

from sklearn import preprocessing
c = preprocessing.normalize(a,norm='l2')

##----- confusion of axis=1 and axis =0 :-
##https://stackoverflow.com/questions/17079279/how-is-axis-indexed-in-numpys-array


# find inverse of a matrix
a = np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3)
np.linalg.inv(a)


################# Random Number with NumPy ####################
a = np.random.randn(1,5) 
#or
a = np.random.randn(5)


# Write a NumPy program to generate six random integers between 5 and 100.
a = np.random.randint(low=5,high=100, size=6)

#Write a NumPy program to create a 3x3x3 array of radnom values
a= np.random.rand(3,3,3)


# find minimum and maximum value in an array
a = np.random.randn(3,3)
a.min()
a.max()

#sort an array (row wise)
a.sort()
a

# Write a NumPy program to create random vector of size 15 and replace the maximum value by -1.

a = np.random.randn(15)
a[a.argmax()] = -1


# Get the 2 largest values of an array
a = np.random.randn(10)
a[np.argsort(a)[-2::]]





