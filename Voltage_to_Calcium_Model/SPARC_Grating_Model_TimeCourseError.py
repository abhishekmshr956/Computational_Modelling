#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:27:23 2021

@author: amishra
"""

#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import Model_functions as model 

#read data and store in pandas dataframe
#Bulle SPARC_T4cGCamp Gratings data
Bulle_SPARC_T4cGcamp_grating_data = pd.read_csv("data/Bulle/Sparc_GCaMP_Grating_TimeSeries.txt",sep="\t")
#entering column name into pandas dataframe
column = ['time']
velocity = [15, 30, 60, 120]
[column.extend(['Mean_PD_'+str(velocity[i]),'SEM_PD_'+str(velocity[i])]) for i in range(len(velocity))];
#[column.extend(['Mean_ND_'+str(velocity[i]),'SEM_ND_'+str(velocity[i])]) for i in range(len(velocity))];
Bulle_SPARC_T4cGcamp_grating_data.columns = column

#plot data and save figure
f, ax = plt.subplots(1,4, sharey=True, figsize=(15,5))
[ax[i].plot(Bulle_SPARC_T4cGcamp_grating_data['Mean_PD_'+str(velocity[i])], label='SPARC_GCamp_PD') for i in range(4)];
#[ax[i].plot(Bulle_T4cGcamp_grating_data['Mean_ND_'+str(velocity[i])], color='red', label='GCamp_ND') for i in range(4)];
plt.legend();
plt.savefig('figures/Bulle_SPARC_Grating.pdf', dpi = 1000)

#Bulle T4cArcLight gratings data
Bulle_arclight_grating_data = pd.read_csv("data/Bulle/ArcLight_Grating_TimeSeries.txt",sep="\t")
column = ['time']
velocity = [15, 30, 60, 120]
[column.extend(['Mean_PD_'+str(velocity[i]),'SEM_PD_'+str(velocity[i])]) for i in range(len(velocity))];
[column.extend(['Mean_ND_'+str(velocity[i]),'SEM_ND_'+str(velocity[i])]) for i in range(len(velocity))];
Bulle_arclight_grating_data.columns = column

#plot data and save figure
f, ax = plt.subplots(1,4, sharey=True, figsize=(15,5))
[ax[i].plot(Bulle_arclight_grating_data['Mean_PD_'+str(velocity[i])], label='Arclight_PD') for i in range(4)];
[ax[i].plot(Bulle_arclight_grating_data['Mean_ND_'+str(velocity[i])], color='red', label='Arclight_ND') for i in range(4)];
plt.legend();
plt.savefig('figures/Bulle_ArcLightGrating.pdf', dpi = 1000)

df_arclight = Bulle_arclight_grating_data.copy() #arclight data copy
dt = df_arclight.iloc[7]['time'] - df_arclight.iloc[6]['time'] #time step dt

#stimulus used for error calculation
velocity = [15, 30, 60, 120]; stim=[]
#[stim.extend(['Mean_PD_'+str(velocity[i]),'Mean_ND_'+str(velocity[i])]) for i in range(len(velocity))];
[stim.extend(['Mean_PD_'+str(velocity[i])]) for i in range(len(velocity))];
print (f'Stimulus used for error calculation {stim}')

#may use less number of parameter values to search through
thres = np.linspace(-0.04, 0.08, 10) #threshold value search
tau =np.linspace(0.05,2.5,10)  #time constants
gain = np.linspace(30.0, 100.0, 10)  #gain
error = np.zeros((len(thres), len(tau), len(gain))) #array to store error values
for i in range(len(thres)):
    for j in range(len(tau)):
        for k in range(len(gain)):
            err_= 0
            for sti in stim:
                PD_Gcamp_data = Bulle_SPARC_T4cGcamp_grating_data[sti].copy()
                PD_arclight = df_arclight[sti].copy()
                PD_Gcamp_model = pd.Series(model.Ca_model(PD_arclight.copy(), tau[j], dt, thres[i], gain[k]))
                shift = model.shift_signal(PD_Gcamp_model, PD_Gcamp_data, float(sti[-2:]))
                PD_Gcamp_model.index = PD_Gcamp_model.index - shift
                err_ += mean_squared_error(PD_Gcamp_model.loc[20:90].values, PD_Gcamp_data.loc[20:90].values)
            error[i, j, k] = err_/len(stim)

print(f'Minimum error {error.min()}')

#parameters corresponding to minimum error
thres_model = thres[np.argwhere(error == np.min(error))[0][0]]
tau_model = tau[np.argwhere(error == np.min(error))[0][1]]
gain_model = gain[np.argwhere(error == np.min(error))[0][2]]
print("Parameters values corresponding to minimum error : ")
print(f'threshold : {thres_model},time constant : {tau_model}, gain:{gain_model}')

#print final figure using model parameters corresponding to minimum error
#also shift and align the signals
velocity = [15, 30, 60, 120]
f, ax = plt.subplots(1,4, sharey=True, figsize=(15,5))
for i in range(len(stim)):
    sti=stim[i]
    PD_Gcamp_data = Bulle_SPARC_T4cGcamp_grating_data[sti].copy()
    PD_arclight = df_arclight[sti].copy()
    PD_Gcamp_model = pd.Series(model.Ca_model(PD_arclight.copy(), tau_model, dt, thres_model, gain_model))
    shift = model.shift_signal(PD_Gcamp_model, PD_Gcamp_data, float(sti[-2:]))
    PD_Gcamp_model.index = PD_Gcamp_model.index - shift
    ax[i].plot(PD_Gcamp_data,'b', label='SPARCGcamp_PD_expt_data')
    ax[i].plot(PD_Gcamp_model, 'b--', label='Gcamp_PD_model', alpha=0.8)
plt.legend();
plt.savefig('figures/SPARC_Grating_Model.pdf', dpi = 1000)



































