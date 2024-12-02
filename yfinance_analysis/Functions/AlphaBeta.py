#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 22:19:54 2024

@author: arnold
"""
from matplotlib import pyplot as plt
import numpy as np

def AlphaBeta(strategy, market, plot=False):
    x, y = market, strategy
    
    coefficients = np.polyfit(x, y, deg=1)
    
    beta = coefficients[0]
    beta = np.round(beta, decimals=3)
    
    alpha = coefficients[1]
    alpha = np.round(alpha, decimals=3)
    
    if plot:
        y_pred = np.polyval(coefficients, x)
        
        plt.scatter(x, y, label='Original Data')
        plt.plot(x, y_pred, color='red', label='Linear Regression')
        text = f'Alpha: {alpha:.2f}\nBeta: {beta:.2f}'
        plt.text(0.2, 0.8, text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlabel('Market Return')  # Changed label
        plt.ylabel('Strategy Return')  # Changed label
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return alpha, beta