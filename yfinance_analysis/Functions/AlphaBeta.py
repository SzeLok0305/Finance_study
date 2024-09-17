#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 22:19:54 2024

@author: arnold
"""
from matplotlib import pyplot as plt
import numpy as np

def AlphaBeta(strategy,market,risk_free_rate,plot=False):
    
    x,y = strategy, market
    
    
    coefficients = np.polyfit(x, y, deg=1)
    
    beta = coefficients[0]
    beta = np.round(beta,decimals=3)
    
    alpha = coefficients[1]
    alpha = np.round(alpha,decimals=3)
    if plot:
        
        y_pred = np.polyval(coefficients, x)
    
        plt.scatter(x, y, label='Original Data')
        plt.plot(x, y_pred, color='red', label='Linear Regression')
        plt.xlabel('Strategy')
        plt.ylabel('Control')
        plt.legend()
        plt.show()
    return alpha, beta