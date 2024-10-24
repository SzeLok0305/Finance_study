import numpy as np
import pandas as pd
import datetime

def MC_simulation(df,step,N_walk):
    D_X1 = (df/df.shift(1)).dropna()
    last_date = df.index[-1]
    date_time = [last_date+datetime.timedelta(j) for j in range(step+1)]
    sim_pd = pd.DataFrame()
    Walks = []
    t = range(step)
    for _ in range(N_walk):
        walk = Random_Walk(D_X1, step)
        Walks.append(walk)
    sim_pd = sim_pd*df.iloc[-1]
    Walks = np.array(Walks)
    Walks = np.cumprod(Walks,axis=1)
    sim_pd['walk_mean'] = Walks.mean(axis=0)*df.iloc[-1]
    sim_pd['1_sigma_lower'] = np.quantile(Walks, 0.159, axis=0) * df.iloc[-1]
    sim_pd['1_sigma_upper'] = np.quantile(Walks, 0.841, axis=0) * df.iloc[-1]
    sim_pd['2_sigma_lower'] = np.quantile(Walks, 0.023, axis=0) * df.iloc[-1]
    sim_pd['2_sigma_upper'] = np.quantile(Walks, 0.977, axis=0) * df.iloc[-1]

    sim_pd['Date'] = date_time
    sim_pd = sim_pd.set_index('Date')
    return sim_pd

def Random_Walk(Time_series_return, step):
    # Generate random samples based on the histogram
    num_steps = step  # Define the number of steps in the random walk
    walk = np.random.choice(Time_series_return, size=num_steps, replace=True)
    walk = np.insert(walk, 0, 1)
    return walk
