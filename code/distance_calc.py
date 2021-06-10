# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:17:12 2021

@author: jburke
"""

from math import sqrt

#####################################################################################################
################################                Euclidean                 ###########################
#####################################################################################################

# Calculate the distance between two points using Euclidean geometry
def calc_euclidean_distance(a, b):
    distance = 0.0
    for i in range(len(a)-1):
        distance += (a[i] - b[i])**2
        pass
    return sqrt(distance)

#####################################################################################################
################################                Hamming                   ###########################
#####################################################################################################

# Calculate the Hamming distance between two arguements 
def calc_hamming_distance(a,b):
    x = a ^ b
    setBits = 0
 
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
    #n = len(a)
    #return manhattan_distancesum(a, n) + manhattan_distancesum(b, n)
    return 0

#####################################################################################################
################################                Mahalanobis                 #########################
#####################################################################################################

def calc_mahalanobis_distance():
    return 0


