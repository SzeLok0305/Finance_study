#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:08:43 2024

@author: arnold
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
import altair as alt
import pandas as pd
import plotly.express as px

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price

st.title('Black-Scholes Option Price Calculator')

# Input parameters
S = st.number_input('Stock Price', min_value=0.01, value=100.0)
K = st.number_input('Strike Price', min_value=0.01, value=80.0)
T = st.slider('Time to Maturity (in years)', min_value=0.1, max_value=3.0, value=0.5, step = 0.1)
r = st.slider('Risk-free Rate', min_value=0.0, max_value=0.20, value=0.05, step = 0.01)
sigma = st.slider('Volatility', min_value=0.01, max_value=1.0, value=0.2,step = 0.01)

# Calculate option prices
call_price = black_scholes(S, K, T, r, sigma, 'call')
put_price = black_scholes(S, K, T, r, sigma, 'put')

# Display results
st.subheader('Option Prices')
#st.write(f'Call Option Price: ${call_price:.2f}')
#st.write(f'Put Option Price: ${put_price:.2f}')
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-around;">
        <div style="background-color: green; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <p style="margin: 0; font-size: 20px;">Call</p>
            <p style="margin: 0; font-size: 24px;">${call_price:.2f}</p>
        </div>
        <div style="background-color: red; color: white; padding: 10px; border-radius: 5px;">
            <p style="margin: 0; font-size: 20px;">Put</p>
            <p style="margin: 0; font-size: 24px;">${put_price:.2f}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

r_SK = np.linspace(0,2,200)
SP = np.linspace(1,100,200)

#T_plot = st.number_input('Time to Maturity (in years)', min_value=0.01, value=1.0)
#r_plot = st.slider('Risk-free Rate', min_value=0.0, max_value=1.0, value=0.05, step = 0.01)
#sigma_plot = st.slider('Volatility', min_value=0.01, max_value=1.0, value=0.2,step = 0.01)

data = pd.DataFrame({
    'S/K': r_SK,
    'Call Price': black_scholes(100, 100*r_SK, 1.0, 0.05, 0.2, 'call'),
    'Put Price': black_scholes(100, 100*r_SK, 1.0, 0.05, 0.2, 'put')
})

data['Call Price'] = black_scholes(SP, r_SK*SP, T, r, sigma, 'call')
data['Put Price'] = black_scholes(SP, r_SK*SP, T,  r, sigma, 'put')

fig_call = px.line(data, x='S/K', y='Call Price', title='call price')
fig_call.update_layout(title={'text': '<b style="text-align: center;">Call Price</b>', 'x': 0.5})

fig_put = px.line(data, x='S/K', y='Put Price', title='put price')
fig_put.update_layout(title={'text': '<b style="text-align: center;">Put Price</b>', 'x': 0.5})

# Display the interactive plot using st.pyplot()
st.plotly_chart(fig_call)
st.plotly_chart(fig_put)

