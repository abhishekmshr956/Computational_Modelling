#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:16:37 2021

@author: amishra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from sklearn.metrics import mean_squared_error

def load_data(filename, column):
    temp = scipy.io.loadmat(filename)
    data = pd.DataFrame(temp['PDmean'])
    data.columns = column
    return data

def column_name(orientation = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], 
                velocity = [15, 30, 60, 120]):
    column = [str(ori)+'_'+str(vel) for ori in orientation for vel in velocity]
    return column

def plot_data(df_data, title, dt=0.0769):
    df_data.index = df_data.index*dt
    df_data.plot(subplots=True, layout=(12, 4), figsize=(20, 20), sharey=True);
    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('figures/'+title+'.pdf',dpi=1000);
    
def lowpass_filter(x, tau=0.550,dt=0.01):
    y = np.zeros(len(x))
    alpha = dt / (tau+dt)
    y[0] = x[0]
    for i in np.arange(1,len(y)):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
        
    return y

def threshold_cut(x, thres=0):
    x[x<thres] = 0
    return x

def Ca_model(x, tau, dt, thres, gain):
    x_thres = threshold_cut(x, thres)
    x_thres_lowpass = lowpass_filter(x_thres, tau, dt)
    x_thres_lowpass_gain = gain*x_thres_lowpass
    
    return x_thres_lowpass_gain

def shift_signal_peak(x_model, x_data, vel):
    if vel == 15.0:
        shift = x_model.iloc[60:90].idxmax() - x_data.iloc[60:90].idxmax()
        return shift
    elif vel == 30.0:
        shift = x_model.iloc[70:90].idxmax() - x_data.iloc[70:90].idxmax()
        return shift
    elif vel == 60.0:
        shift = x_model.iloc[25:37].idxmax() - x_data.iloc[25:37].idxmax()
        return shift
    elif vel == 20.0: #for 120.0 deg/sec
        #shift=0
        shift = x_model.iloc[25:45].idxmax() - x_data.iloc[25:45].idxmax()
        return shift
    
def shift_signal_correlate(x_model, x_data):
    shift = np.argmax(signal.correlate(x_model, x_data)) - (len(x_model) - 1)
    return shift