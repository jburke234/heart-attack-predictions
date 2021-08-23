# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:17:12 2021

@author: jburke
"""
import numpy as np
from math import sqrt

#####################################################################################################
################################                Euclidean                 ###########################
#####################################################################################################

# Calculate the distance between two points using Euclidean geometry
def calc_euclidean_distance(a, b):
    distance = 0.0

    for i in range(len(a)):
        distance += (a[i] - b[i])**2
        
    return sqrt(distance)

#####################################################################################################
################################                Hamming                   ###########################
#####################################################################################################

# Calculate the Hamming distance between two arguements 
def calc_hamming_distance(a,b):
    setBits = 0
    
    for i in range(len(a)):
        x = a[i] ** b[i]
        
        while (x > 0) :
            setBits += x & 1
            x >>= 1
            
    return setBits

#####################################################################################################
################################                Manhattan                 ###########################
#####################################################################################################

def manhattan_distancesum (arr, n):  
    arr.sort()
     
    res = 0
    sum = 0
    for i in range(n):
        res += (arr[i] * i - sum)
        sum += arr[i]
     
    return res   

def calc_manhattan_distance(a, b):
    n = len(a)
    return manhattan_distancesum(a, n) + manhattan_distancesum(b, n)


#####################################################################################################
################################                Mahalanobis                 #########################
#####################################################################################################

def calc_mahalanobis_distance(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    #x_minus_mu = x - np.mean(data)
    #if not cov:
    #    cov = np.cov(data.values.T)
    #inv_covmat = sp.linalg.inv(cov)
    #left_term = np.dot(x_minus_mu, inv_covmat)
    #mahal = np.dot(left_term, x_minus_mu.T)
    #return mahal.diagonal()
    
    return 0 


#####################################################################################################
################################                Chi-Squared                 #########################
#####################################################################################################

def calc_chi_squared_distance():
    pass


#####################################################################################################
################################                Cosine                      #########################
#####################################################################################################

def calc_cosine_distance():
    pass


#####################################################################################################
################################                Minkowsky                   #########################
#####################################################################################################

def calc_minkowski_distance():
    pass




