# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import os
import warnings
import math
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
#%%

class Model_Simple(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model_Simple, self).__init__()
        
        #types of layer
        hidden_size = input_size
        hidden_size1 = input_size//2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size1)
        self.fcFinal = nn.Linear(hidden_size1, output_size)
        
        #types of activation function
        self.drop = nn.Dropout(p=0.5)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        
        out = x
        out = self.fc1(out)
        out = self.drop(out)
        out = self.sigmod(out)
        
        out = self.fc2(out)
        out = self.drop(out)
        out = self.sigmod(out)
        
        out = self.fcFinal(out)
        out = self.sigmod(out)
        #out = torch.round(out)
        return out


def Train_Simple_Model(stock_ticker,test_data_split):
#Load Data
    df = yf.download(stock_ticker,start="2010-01-01")
    df.dropna(inplace=True)
    df["Open_PerCh"] = df["Open"].pct_change()
    df["Close_PerCh"] = df["Close"].pct_change()
    df["High_PerCh"] = df["High"].pct_change()
    df["Low_PerCh"] = df["Low"].pct_change()
    
    model_name = "simple_up_and_down"
    df["Moving_Average"] = df["Open"].rolling(window=5).mean()
    df.dropna(inplace=True)
    # Assign "Signal" values based on the condition
    df["Signal"] = np.where((df["Open"] > df["Moving_Average"]) & (df["Open_PerCh"] > 0), 1, 0)
    df.dropna(inplace=True)
    
    
    features_columns = ['Open_PerCh', 'Close_PerCh', 'High_PerCh', 'Low_PerCh']
    features = df.loc[:, features_columns].to_numpy()
    
    label = df.loc[:,"Signal"]
    #Check label distribution
    label.hist()
    label = label.to_numpy()
    
    features_train, features_test, label_train, labe_test = train_test_split(features, label, test_size=test_data_split,shuffle=False)
    
    features_train = torch.tensor(features_train).float()
    features_test = torch.tensor(features_test).float()
    label_train = torch.tensor(label_train).view(-1,1).float()
    labe_test = torch.tensor(labe_test).view(-1,1).float()
    
    
    # Instantiate network
    input_size = features_train.shape[1]
    output_size = 1 #Train_label.shape[1]
    model = Model_Simple(input_size,output_size)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize a list to store the loss values
    loss_values = []

    # Set the number of training epochs
    num_epochs = 1500 #Number_of_initial_data

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(features_train)
        loss = criterion(outputs, label_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the loss value for this epoch
        loss_values.append(loss.item())

        # Print the training loss for every few epochs
        if (epoch+1) % 10 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    # Save the trained model weights "Stock_Data/" 
    save_dir = "Stock_Data/" + stock_ticker + "/Simple_model/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    
    # Apply the trained model on features_test
    with torch.no_grad():
        model.eval()
        model_prediction = model(features_test)
        df.loc[df.index[-len(features_test):], "model_prediction"] = model_prediction.numpy()

    # Plotting the error function against the epoch
    plt.plot(range(1, num_epochs+1), loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
    model.eval()
    
    
    return df

df_test = Train_Simple_Model("MMM",0.1)

#%%

class Model_Yahoo(nn.Module):
    def __init__(self, output_size):
        super(Model_Yahoo, self).__init__()
        
        #types of layer
        hidden_size = 10
        hidden_size2 = 5
        self.fcFinal = nn.Linear(hidden_size2, output_size)
        
        self.two_to_two = nn.Linear(2, 2)
        self.two_to_one = nn.Linear(2, 1)
        self.one_to_one = nn.Linear(1, 1)
        self.one_to_two = nn.Linear(1, 2)
        self.correlate_all = nn.Linear(5, hidden_size2)
        self.correlate_to_output = nn.Linear(3, 1)
        
        #types of activation function
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.6)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        # Separate the tensor
        open_yestaday = x[:,1]
        close_yestaday = x[:,2]
        high_yestaday = x[:,3]
        low_yestaday = x[:,4]
        open_today = x[:,5]
        volume_delta = x[:,6]
        
        OY = open_yestaday.view(-1,1)
        CY = close_yestaday.view(-1,1)
        HY = high_yestaday.view(-1,1)
        LY = low_yestaday.view(-1,1)
        OT = open_today.view(-1,1)
        
        def make_2vector(a,b):
            v = np.transpose(np.array((a,b)))
            v = torch.tensor(v).float()
            return v
        
        def combine_2tensor(a,b):
            return torch.cat((a,b)).view(-1,2)
        
        #Firse layer
        OL = make_2vector(open_yestaday,low_yestaday)
        OH = make_2vector(open_yestaday,high_yestaday)
        
        DV = volume_delta.view(-1,1)
        #############################################
        
        #OL = self.two_to_two(OL)
        #OL = self.drop(OL)
        #OL = self.sigmod(OL)
        OL = self.two_to_one(OL)
        OL = self.drop(OL)
        OL = self.sigmod(OL)
        
        #OH = self.two_to_two(OH)
        #OH = self.drop(OH)
        #OH = self.sigmod(OH)
        OH = self.two_to_one(OH)
        OH = self.drop(OH)
        OH = self.sigmod(OH)
        
        OLC = combine_2tensor(OL,CY)
        OLC = self.two_to_one(OLC)
        OLC = self.drop(OLC)
        OLC = self.sigmod(OLC)
        
        OHC = combine_2tensor(OH,CY)
        OHC = self.two_to_one(OHC)
        OHC = self.drop(OHC)
        OHC = self.sigmod(OHC)
        
        OLCO = combine_2tensor(OLC,OT)
        OLCO = self.two_to_one(OLCO)
        OLCO = self.drop(OLCO)
        OLCO = self.sigmod(OLCO)
        
        OHCO = combine_2tensor(OHC,OT)
        OHCO = self.two_to_one(OHCO)
        OHCO = self.drop(OHCO)
        OHCO = self.sigmod(OHCO)
        
        DV = self.one_to_one(DV)
        DV = self.drop(DV)
        DV = self.sigmod(DV)
        
        
        Combined = torch.cat((OLCO,OHCO,DV), dim=1)
        out = self.correlate_to_output(Combined)
        out = self.drop(out)
        out = self.sigmod(out)
        return out
    