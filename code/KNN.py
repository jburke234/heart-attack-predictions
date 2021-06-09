# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:33:11 2021

@author: jburke
@description: 
    An Implementation of K-Nearest Neighbours (KNN) to predict Heart Attacks
    
    In KNN when a new data point is submitted the closest existing datapoints in the training set are located 
    These data points are then used to make a prediction for the new data point.
    
    To achieve this we can  calculate the distances between each of the data points in the 
    training set. It is also used to find the nearest neighbour to the new data point. 
    
    The prediction for the new data point is made using the average result of the nearest neighbours.
            
"""
# Imports
import system as sys
import distance_calc.py as calc


################################################################################################
###################                    KNN Implementation                    ###################
################################################################################################

# To find the relationship between a new data point and the existing data points, we calculate
# the distance between the new data point and all of the existing data points.
def find_neighbors(train, test_row, num_neighbors, calc_method):
    distances = list()
    for train_row in train:
        if(calc_method == 0):
             dist = calc.calc_euclidean_distance(test_row, train_row)
        elif():
            dist = calc.calc_hamming_distance()
        elif():
            dist = calc.calc_manhattan_distance()
        else:
            print("No distance calculation selected - EXITING")
            sys.exit()
        distances.append((train_row, dist))
        
    # To find the nearest neighbours to the new data point we sort the calculated distance, to find 
    # the nearest existing data points    
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = find_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction


