# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:33:21 2021

@author: jburke

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

def generate_folds(k):
    kfold = KFold(k, True, 1)
    return kfold
       
    
def main():
    df = open_dataset()
    
    # Need to find correlation between features and output
    
    # Features 
    X = df.iloc[:, [0, 12]]
    
    # Label
    y = df.iloc[:, 13]
    
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    num_neighbors = 5
    
    calc_method = 0
    
    # Loop through a range of k values to evaluate the number of folds accuracy for each model.
    for k in range(2,21):
        kfold = generate_folds(k)
        for train, test in kfold.split(X):
            y_pred = knn.predict_classification(train, test, num_neighbors, calc_method)
            
    
    #train, test =             
            
            
    # Resulting output for each distance calculation  should be:
    #             num_neighbors    
    # num_folds |      1      |   
    #    2      | Acc_Score   |



if __name__ == '__main__':
    main()