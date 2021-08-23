# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:33:21 2021

@author: jburke

@paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978658/

@data_source: https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

@data_description:
    age         - Age of the individual in years   
    sex         - Sex of the individual (1 = male; 0 = female) 
    cp          - Chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)  
    trtbps      - Resting blood pressure (in mm Hg)  
    chol        - Serum cholestoral in mg/dl   
    fbs         - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) 
    restecg     - Resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy) 
    thalachh    - Maximum heart rate achieved 
    exng        - Exercise induced angina (1 = yes; 0 = no) 
    oldpeak     - ST depression induced by exercise relative to rest  
    slp         - The slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)   
    caa         - Number of major vessels (0-3) colored by flourosopy  
    thall       - Thal rate (normal = 3; fixed defect = 6; reversable defect = 7)  
    output      - The target variable (Label)

"""
# Util imports 
import os
import os.path
import shutil
import pandas as pd
import numpy as np 

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import KNN as knn
import plot as plot

def open_dataset():
    # Get the current working directory
    current_dir = os.getcwd() 
    
    # Go up one directory from working directory
    os.chdir("..") 
    
    # Get the path for the data folder 
    data_path = os.path.join(current_dir, '..\\data')
    
    # Change working directory to the data folder
    os.chdir(data_path) 
    
    # Read the CSV into a DataFrame
    df= pd.read_csv('heart.csv', sep=",", decimal='.' )
    
    os.chdir(current_dir)
    
    return df

def write_to_file(df):
    # Get the current working directory
    current_dir = os.getcwd() 
    
    # Go up one directory from working directory
    os.chdir("..") 
    
    # Get the path for the data folder 
    output_path = os.path.join(current_dir, '..\\output')  
    
    # Change working directory to the data folder
    os.chdir(output_path) 
    
    df.to_csv('Acc_Scores.csv', index = False)
    
    os.chdir(current_dir)

def generate_folds(k):
    kfold = KFold(k, True, 1)
    return kfold
      
    
def main():
    
    # Open the dataset from the data folder
    df = open_dataset()
    
    # Features 
    X = df.iloc[:, [0, 12]]
    
    # Label
    y = df.iloc[:, 13]
    
    # Scale feature data to fit in a range 
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    # List to hold the [number of folds, number of neighbors, accuracy score]    
    overall_scores = list()
    
    # Controls the type of distance calculation done 
    for calc_method in range(0,2):
        
        # Loop through a range of k values to evaluate the number of folds accuracy for each model.
        for k in range(2,31):
            
            kfold = generate_folds(k)
            
            # Loop through a range of neighbor number values 
            for num_neighbors in range(2,31):
                
                print(f"Folds : {k}, Neighbors: {num_neighbors}")
                
                # Holds the accuracy score for each fold 
                scores = []
                
                # Used to calculate the average accuracy score across the folds 
                score = 0
                
                # Loop through each of the folds in the data 
                for train, test in kfold.split(X):
                    
                    test_count = 0
                    correct = 0
                    
                    # Loop through each of the test rows
                    for test_row in test:
                        
                        # Returns the prediction for the test row as a 1 or a 0 
                        y_pred = knn.predict_classification(train, test_row, num_neighbors, calc_method, X, y)     
                        
                        #print(f"Row ID: {test_row}, Expected: {y[test_row]}, Got: {y_pred}")
                        
                        if(y_pred == y[test_row]):
                            correct += 1
                        
                        test_count += 1
                    
                    accuracy = (correct / test_count) * 100
                    
                    scores.append(accuracy)
                        
                        #scores.append(accuracy_metric(y[test_row], y_pred))
                
                score = np.round(np.mean(scores), 2)    
                #print(scores)        
                
                # Append the number of folds, number of neighbors and the resulting accuracy score to the desired scores list 
                if(calc_method == 0):
                    overall_scores.append(["Euclid", k, num_neighbors, score])
                elif(calc_method == 1):
                    overall_scores.append(["Manhattan", k, num_neighbors, score])
                # elif(calc_method == 2):
                #      overall_scores.append(["Hamming", k, num_neighbors, score])
                # elif(calc_method == 3):
                #     overall_scores.append(["Mahalanobis", k, num_neighbors, score])
                # elif(calc_method == 4):
                #     overall_scores.append(["Chi-Squared", k, num_neighbors, score])
                # elif(calc_method == 5):
                #     overall_scores.append(["Cosine", k, num_neighbors, score])
                # elif(calc_method == 6):
                #     overall_scores.append(["Minkowski", k, num_neighbors, score])
    
    
    # Convert the overall data into a Dataframe for visualisation 
    scores_df = pd.DataFrame(overall_scores[0:],columns=["dist_calc_type", "num_folds", "num_neighbors", "acc_score"])
    print(scores_df)
    
    write_to_file(scores_df)
    
    #plot.multiplot(scores_df, "num_folds", "acc_score")
    

if __name__ == '__main__':
    main()