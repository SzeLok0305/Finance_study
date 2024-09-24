import streamlit as st
import numpy as np
from scipy.stats import norm
import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def black_scholes_pricing(S, K, T, r, q, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r - q  + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price

def jump_diffusion_simulation(S0, r, q, sigma, lambda_j, mu_j, sigma_j, T, N, num_sims):
    #Model: dS = (r-q) * S * dt + sigma * S * dW + S * dN(lambda_j) * dJ(mu_j,sigma_j)
    # Remember to use ito calculus  !!!!!!!
    dt = T / N
    paths = np.zeros((num_sims, N+1))
    paths[:, 0] = S0
    
    for i in range(1, N+1):
# The solution take the form: S_t = S_0 * exp( (r - q - 1/2 * sigma^2) * t + sigma * Normal(0,sqrt(t)) + Poisson(lambda_j * dt) * Normal(mu_j,sigma_j) )
        
        # Diffusion component
        dW = np.random.normal(0, np.sqrt(dt), num_sims)
        diffusion = (r - q - 0.5 * sigma**2) * dt + sigma * dW
        
        # Jump component
        dN = np.random.poisson(lambda_j * dt, num_sims)
        J = np.random.normal(mu_j, sigma_j, num_sims)
        jump = J * dN
        
        #Combine
        paths[:, i] = paths[:, i-1] * np.exp(diffusion + jump)
        
    return paths

def jump_diffusion_pricing_sims(Spot, Strike, T, r, q, sigma, lambda_j, mu_j, sigma_j, option_type='call', num_sims=2**13):
    # Simulate stock price paths
    N = int(T*252)
    paths = jump_diffusion_simulation(Spot, r, q, sigma, lambda_j, mu_j, sigma_j, T, N, num_sims)
    
    # Calculate terminal stock prices
    S_T = paths[:, -1]
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(S_T - Strike, 0)
    elif option_type == 'put':
        payoffs = np.maximum(Strike - S_T, 0)
    else:
        print('Choose the either call or put, thank you.')
        return
 
    # Calculate option price
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price

def jump_diffusion_pricing_plot(Spot, Strike, T, r, q, sigma, lambda_j, mu_j, sigma_j, option_type='call', num_sims=2**13):
    prices = np.zeros(len(Spot))
    for i in range(len(Spot)):
        prices[i] = jump_diffusion_pricing_sims(Spot[i], Strike, T, r, q, sigma, lambda_j, mu_j, sigma_j, option_type)
    return prices


def create_heatmap(prices, title, option_type):
    fig = go.Figure(data=go.Heatmap(
        z=prices,
        x=T_range,
        y=sigma_range,
        colorscale='Greys',
        showscale=False
    ))

    # Add text annotations
    for i in range(len(sigma_range)):
        for j in range(len(T_range)):
                if option_type == 'Call':
                    intrinsic_value = max(S_K_heat-1, 0)
                elif option_type == 'Put':
                    intrinsic_value = max(1-S_K_heat, 0)
                time_value = prices[j, i] - intrinsic_value
                if time_value > TV_heat/100 * S_K_heat:
                    text_color = 'Green'
                else:
                    text_color = 'red'
                fig.add_annotation(
                x=T_range[j],
                y=sigma_range[i],
                text=f"{prices[i, j]:.2f}",
                showarrow=False,
                font=dict(size=16, color=text_color)
            )

    fig.update_layout(
        title={
            'text': f'<b>{title}</b>',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Time to maturity',
        yaxis_title="Volatility",
        height=500,
        width=2000
    )

    return fig

def Display_prices(call_price, put_price):
    st.subheader('Option Prices')
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-around;">
            <div style="background-color: green; color: white; padding: 10px; border-radius: 5px; text-align: center;">
                <p style="margin: 0; font-size: 20px;">Call</p>
                <p style="margin: 0; font-size: 24px;">${call_price:.2f}</p>
            </div>
            <div style="background-color: red; color: white; padding: 10px; border-radius: 5px; text-align: center;">
                <p style="margin: 0; font-size: 20px;">Put</p>
                <p style="margin: 0; font-size: 24px;">${put_price:.2f}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    return 


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Black Scholes model", "Jump diffusion model"])

    if page == "Home":
        home_page()
    elif page == "Black Scholes model":
        BS_option_calculator_page()
    elif page == "Jump diffusion model":
        JD_option_calculator_page()

def home_page():
    st.title("Welcome to the Option Pricing App")
    st.write("This app provides tools for European option pricing using different models.")
    st.write("Use the sidebar to navigate to different pages:")

def BS_option_calculator_page():
    st.title('Black-Scholes Option Price Calculator')

    # Input parameters
    S = st.number_input('Stock Price', min_value=0.01, value=100.0)
    K = st.number_input('Strike Price', min_value=0.01, value=80.0)
    T = st.slider('Time to Maturity (in years)', min_value=0.1, max_value=3.0, value=0.5, step = 0.01)
    r = st.slider('Risk-free Rate(%)', min_value=0.0, max_value=20.0, value=5.0, step = 0.5)
    q = st.slider("Dividend(%)", min_value=0.0, max_value=20.0, value=3.0, step = 0.5)
    sigma = st.slider('Volatility(%)', min_value=0.0, max_value=100.0, value=20.0,step = 1.0)

    # Calculate option prices
    call_price = black_scholes_pricing(S, K, T, r/100, q/100, sigma/100, 'call')
    put_price = black_scholes_pricing(S, K, T, r/100, q/100, sigma/100, 'put')

    Display_prices(call_price,put_price)

    r_SK = np.linspace(0,2,200)

    data = pd.DataFrame({
        'S/K': r_SK,
        'C/K': black_scholes_pricing(r_SK, 1, 1.0, 1.0, 0.05, 0.2, 'call'),
        'P/K': black_scholes_pricing(r_SK, 1, 1.0, 1.0, 0.05, 0.2, 'put')
    })

    data['C/K'] = black_scholes_pricing(r_SK, 1, T, r/100, q/100, sigma/100, 'call')
    data['P/K'] = black_scholes_pricing(r_SK, 1, T,  r/100, q/100, sigma/100, 'put')

    fig_call = px.line(data, x='S/K', y='C/K', title='call price')
    fig_call.update_layout(title={'text': '<b style="text-align: center;">Call Price/Stock Price </b>', 'x': 0.5})

    fig_put = px.line(data, x='S/K', y='P/K', title='put price')
    fig_put.update_layout(title={'text': '<b style="text-align: center;">Put Price/Stock Price</b>', 'x': 0.5})

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_call, use_container_width=True)
    with col2:
        st.plotly_chart(fig_put, use_container_width=True)  

def JD_option_calculator_page():
    st.title('Jump diffusion Option Price Calculator')

    # Input parameters
    S = st.number_input('Stock Price', min_value=0.01, value=100.0)
    K = st.number_input('Strike Price', min_value=0.01, value=80.0)
    T = st.slider('Time to Maturity (in years)', min_value=0.1, max_value=3.0, value=0.5, step = 0.01)
    r = st.slider('Risk-free Rate(%)', min_value=0.0, max_value=20.0, value=5.0, step = 0.5)
    q = st.slider("Dividend(%)", min_value=0.0, max_value=20.0, value=3.0, step = 0.5)
    sigma = st.slider('Volatility(%)', min_value=0.0, max_value=100.0, value=20.0,step = 1.0)
    lambda_j = st.slider('Jump intensity (average number of jump per year)', min_value=0.0, max_value=10.0, value=5.0,step = 0.50)
    mu_j = st.slider('mean jump size (%)', min_value=-100.0, max_value=100.0, value=0.0,step = 5.0)
    sigma_j = st.slider('Volatility of jump size (%)', min_value=0.0, max_value=100.0, value=20.0,step = 5.0)

    # Calculate option prices
    call_price = jump_diffusion_pricing_sims(S, K, T, r/100, q/100, sigma/100, lambda_j/100, mu_j/100, sigma_j/100, 'call')
    put_price = jump_diffusion_pricing_sims(S, K, T, r/100, q/100, sigma/100, lambda_j/100, mu_j/100, sigma_j/100, 'put')

    Display_prices(call_price,put_price)

    #r_SK = np.linspace(0,2,200)

    #data = pd.DataFrame({
    #    'S/K': r_SK,
    #    'C/K': jump_diffusion_pricing_plot(r_SK, 1, 1.0, 0.05, 0.02, 0.10, 0.10, 0.10, 0.10, 'call'),
    #    'P/K': jump_diffusion_pricing_plot(r_SK, 1, 1.0, 0.05, 0.02, 0.10, 0.10, 0.10, 0.10, 'put')
    #})

    #data['C/K'] = jump_diffusion_pricing_plot(r_SK, 1, T, r/100, q/100, sigma/100, lambda_j/100, mu_j/100, sigma_j/100, 'call')
    #data['P/K'] = jump_diffusion_pricing_plot(r_SK, 1, T,  r/100, q/100, sigma/100, lambda_j/100, mu_j/100, sigma_j/100, 'put')

    #fig_call = px.line(data, x='S/K', y='C/K', title='call price')
    #fig_call.update_layout(title={'text': '<b style="text-align: center;">Call Price/Stock Price </b>', 'x': 0.5})

    #fig_put = px.line(data, x='S/K', y='P/K', title='put price')
    #fig_put.update_layout(title={'text': '<b style="text-align: center;">Put Price/Stock Price</b>', 'x': 0.5})

    #col1, col2 = st.columns(2)
    #with col1:
    #    st.plotly_chart(fig_call, use_container_width=True)
    #with col2:
    #    st.plotly_chart(fig_put, use_container_width=True)  

if __name__ == "__main__":
    main()