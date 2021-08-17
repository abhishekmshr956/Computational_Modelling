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

