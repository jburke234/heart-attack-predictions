# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:04:39 2021

@author: jburke
"""

import seaborn as sns 


def multiplot(data, r, c):
    g = sns.FacetGrid(data, row=r, col=c, margin_titles=True)
    g.map(sns.regplot, "folds", "accuracy", color=".3", fit_reg=False, x_jitter=.1)