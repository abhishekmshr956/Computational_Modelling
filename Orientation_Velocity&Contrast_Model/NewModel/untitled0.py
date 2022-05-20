#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:22:40 2021

@author: amishra
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def load_data(filename):
    """function to load data"""
    x = scipy.io.loadmat(filename)['PDmean']
    return x

global T4_Arclight, T4Gcamp #global varaiables to hold Arclight and Gcamp data

filename = 'data/T4c_ArcLight/ARCOri_Velocity_Mean.mat' 
T4_Arclight = load_data(filename)  #load Arclight data

filename = 'data/T4c_GCaMP/Ori_Mean_Velocity.mat'
T4_Gcamp = load_data(filename)  #load Gcamp data

global p_bounds #bounds on parameters (thres, tauhp, taulp1, gain1, taulp2, gain2, tshift)
p_bounds = ((-1,1),(0,10),(0,10),(0,200),(0,10),(0,200),(-113,113))

def plot_data(data_list, dt=0.0769, fig_size=(20,20), speed=[15,30,60,120], title='', savefig=False, c=['k','r']):
    """Plots data for multiple datsets"""
    n_col = len(speed) #number of columns i.e. number of speed stimuli
    n_sti = data_list[0].shape[1] #number of orientations
    x_values = np.arange(data_list[0].shape[0]) * dt #changing x-axis to time. multiply with time step dt = 1/frequency
    f, ax = plt.subplots(int(n_sti/n_col), int(n_col), sharex=True, sharey= True, figsize=fig_size)
    c_count = 0 #color count for different dataset
    for data in data_list:
        count = 0 #count for stimuli
        for i in range(int(n_sti/n_col)):
            for j in range(n_col):
                ax[i, j].plot(x_values, data[:, count], color=c[c_count])
                count = count + 1
        c_count += 1
    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    if savefig:
        plt.savefig('figures/'+title+'.pdf',dpi=1000);

def plot_data_twinaxis(data_list, dt=0.0769, fig_size=(20,20), speed=[15,30,60,120], title='', savefig=False, c=['k','r']):
    """Plots data for multiple datsets"""
    n_col = len(speed) #number of columns i.e. number of speed stimuli
    n_sti = data_list[0].shape[1] #number of orientations
    x_values = np.arange(data_list[0].shape[0]) * dt #changing x-axis to time. multiply with time step dt = 1/frequency
    f, ax = plt.subplots(int(n_sti/n_col), int(n_col), sharex=True, sharey= True, figsize=fig_size)
    c_count = 0 #color count for different dataset
    for data in data_list:
        count = 0 #count for stimuli
        for i in range(int(n_sti/n_col)):
            for j in range(n_col):
                if c_count == 0:
                    ax[i, j].plot(x_values, data[:, count], color=c[c_count])
                else :
                    ax1 = ax[i,j].twinx()
                    ax1.plot(x_values, data[:, count], color=c[c_count])
                #ax[i, j].plot(data[:, count], color=c[c_count])
                count = count + 1
        c_count += 1
    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    if savefig:
        plt.savefig('figures/'+title+'.pdf',dpi=1000);
        
#plot_data([T4_Arclight]) 

def lowpass(x, tau, dt):
    y = np.zeros_like(x)
    n = x.shape[0]   #length of time dimension
    alpha = dt / (tau+dt)
    y[0] = x[0]
    for i in np.arange(1,n):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    return y

def highpass(x, tau, dt):
    y = x - lowpass(x, tau, dt)
    return y

def bandpass(x, tauhp, taulp, dt):
    y = highpass(x, tauhp, dt)
    y = lowpass(y, taulp, dt)
    return y

def threshold_cut(x, thres):
    x_copy = x.copy()
    x_copy[x_copy<thres] = 0
    return x_copy 

def Ca_model(x, p, dt=0.0769):
    """x is Arclight data. p is list of parameters. dt is timestep"""
    (thres, tauhp, taulp1, gain1, taulp2, gain2, tshift) = p
    x_thres = threshold_cut(x, thres)
    x_thres_bp1 = bandpass(x_thres, tauhp, taulp1, dt)
    x_1 = x_thres_bp1 * gain1
    x_thres_bp2 = bandpass(x_thres, tauhp, taulp2, dt)
    x_2 = x_thres_bp2 * gain2
    y = x_1 + x_2
    y = np.roll(y, int(tshift), axis=0)
    return y

def create_random_params():
    p = [np.random.uniform(bounds[0], bounds[1]) for bounds in p_bounds]
    return p  

def calc_error(p):
    T4Ca_model = Ca_model(T4_Arclight, p)
    error1 = np.sqrt(np.mean((T4_Gcamp-T4Ca_model)**2)) #timeerror
    error2 = np.sqrt(np.mean((T4_Gcamp.max(axis=0)-T4Ca_model.max(axis=0))**2)) #peakerror
    timeerror = 1.0 #weight for time error
    error = timeerror*error1 + (1.0-timeerror)*error2 #peak error aand time error combined
    return error

def fit_params():
    p = create_random_params() #creates random parameter values
    #options = {'maxiter':5000} #maximum number of iterations
    res = minimize(calc_error, p, tol=1e-10, bounds=p_bounds)#options=options
    if res.success:
        print('Optimisation successful')
    else:
        print('Optimisation not succesfull')
    p = res.x
    print(f'Remaining error:{calc_error(p)}')
    print(res)
    return p

model_p = fit_params()
T4_model = Ca_model(T4_Arclight, model_p)
#plot_data([T4_Gcamp, T4_model])
        
    


    

    
        
        
        
        
        
        
        
        
        
        
        
        
















    
        