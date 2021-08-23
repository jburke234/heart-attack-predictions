# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:04:39 2021

@author: jburke
"""
import os
import os.path

import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
from mpl_toolkits.mplot3d import Axes3D

def multiplot(data, r, c):
    g = sns.FacetGrid(data, row=r, col=c, margin_titles=True)
    g.map(sns.regplot, "folds", "accuracy", color=".3", fit_reg=False, x_jitter=.1)
    

def open_output_file():
    # Get the current working directory
    current_dir = os.getcwd() 
    
    # Go up one directory from working directory
    os.chdir("..") 
    
    # Get the path for the data folder 
    output_path = os.path.join(current_dir, '..\\output')  
    
    # Change working directory to the data folder
    os.chdir(output_path) 
    
    # Read the CSV into a DataFrame
    df= pd.read_csv('Acc_Scores.csv', sep=",", decimal='.' )
    
    os.chdir(current_dir)
    
    return df
  
def generate_3D_plot(df, title):
    
    sns.set(style="darkgrid")
    
    fig = plt.figure()
    
    
    
    ax = fig.add_subplot(111, projection='3d')
    
    #ax = Axes3D(fig)
    
    x = df['num_folds']
    
    y = df['num_neighbors']
    
    z = df['acc_score']
    
    ax.set_xlabel("Folds")
    ax.set_ylabel("Neighbors")
    ax.set_zlabel("Accuracy")
    
    ax.set_title(title)
    
    ax.scatter(x,y,z, c=z, cmap='jet', linewidth=0.5)
    
    
    plt.show()

def generate_2D_trends(df, x_string , y_string):
    sns.set(style="darkgrid")
    
    fig = plt.figure()
    
    sns.scatterplot(data=df, x = x_string, y = y_string, hue="num_folds")
    
    plt.show()

def generate_grid(df):
    
    grid = sns.FacetGrid(df, col='num_folds', hue='dist_calc_type', col_wrap=5)
    
    grid.map(sns.lineplot, "num_neighbors", "acc_score")
    
    grid.add_legend()

def generate_plots(df):
    
    matplotlib.rc('figure', figsize=(12,6))
    
    # For test purposes
    # df = open_output_file()
    
    dist_calc_type = df['dist_calc_type'].unique()
    
    for dist_type in dist_calc_type:
        generate_3D_plot(df.loc[df['dist_calc_type'] == dist_type], dist_type)
    
    # for dist_type in dist_calc_type:
    #     generate_2D_trends(df.loc[df['dist_calc_type'] == dist_type], "num_neighbors", "acc_score")
    
    # generate_2D_trends(df, "num_neighbors", "acc_score")
    
    # generate_2D_trends(df, "num_folds", "acc_score")
    
    generate_grid(df)
    