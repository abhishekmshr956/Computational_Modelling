#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:24:13 2021

@author: amishra
"""
import numpy as np

def lowpass_filter(x, tau=0.550,dt=0.01):
    """
function for implementing low pass filter
Parameters : array, time constant, dt
    """
    y = np.zeros(len(x))
    alpha = dt / (tau+dt)
    y[0] = x[0]
    for i in np.arange(1,len(y)):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
        
    return y

def threshold_cut(x, thres=0):
    """ function for threshold cut
    Parameters : array, threshold"""
    x[x<thres] = 0
    return x

def Ca_model(x, tau=0.550, dt=0.0769, thres=0.0, gain=1.0):
    """function for generating calcium model responses. 
    Parameters : array, lowpass filter time constant, dt,
    threshold, gain
    Steps : First threshold the signal, lowpass this signal,
    multiply gain to this signal"""
    x_thres = threshold_cut(x, thres)
    x_thres_lowpass = lowpass_filter(x_thres, tau, dt)
    x_thres_lowpass_gain = gain*x_thres_lowpass
    
    return x_thres_lowpass_gain

def peak_error_calc(x, y):
    """calculates square of error between the maximum value of two signals"""
    err = (np.nanmax(x) - np.nanmax(y))**2
    return err

def shift_signal(x_model, x_data, vel):
    """function used to return shift value needed to align the gratings signal. 
    It locates either the last or first peak and returns the shift value """
    if vel == 15.0:
        shift = x_model.iloc[60:90].idxmax() - x_data.iloc[60:90].idxmax() #aligns last peak
        return shift
    elif vel == 30.0:
        shift = x_model.iloc[70:90].idxmax() - x_data.iloc[70:90].idxmax() #aligns last peak
        return shift
    elif vel == 60.0:
        shift = x_model.iloc[25:37].idxmax() - x_data.iloc[25:37].idxmax() #aligns first peak
        return shift
    elif vel == 20.0: #for 120.0 deg/sec
        #shift=0
        shift = x_model.iloc[25:45].idxmax() - x_data.iloc[25:45].idxmax() #aligns first peak
        return shift

