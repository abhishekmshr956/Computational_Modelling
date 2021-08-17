#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:56:36 2021

@author: amishra
"""
#import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import glob as glob
import os
import Model_functions as model

#read data and store in pandas dataframe
#Bulle T4cArclight ON Edge data
Bulle_arclight_ON_Edge_data = pd.read_csv("data/Bulle/ArcLight_Edge_TimeSeries.txt",sep="\t")
#entering column names into pandas dataframe
column = ['time']
velocity = [15, 30, 60, 120, 240]
[column.extend(['Mean_PD_'+str(velocity[i]), 'SEM_PD_'+str(velocity[i]),
               'Mean_ND_'+str(velocity[i]), 'SEM_ND_'+str(velocity[i])]) for i in range(len(velocity))];
Bulle_arclight_ON_Edge_data.columns = column

#plotting the data and saving figure
f, ax = plt.subplots(1,5, sharey=True, figsize=(15,5))
[ax[i].plot(Bulle_arclight_ON_Edge_data['Mean_PD_'+str(velocity[i])],'b', label='PD_Arclight') for i in range(5)];
[ax[i].plot(Bulle_arclight_ON_Edge_data['Mean_ND_'+str(velocity[i])], 'r', label='ND_Arclight') for i in range(5)];
plt.legend();
plt.savefig('figures/Bulle_ArcLight_ONEdge.pdf', dpi = 1000)

#Abhishek SPARC_GCamp ON Edge data
#join individual pandas dataframes into one dataframe
DATAPATH = os.path.join(os.getcwd(),'data/Abhishek/SPARC/ONEdge/')
df_list = os.listdir(DATAPATH)
df_SPARC_Abhi_ONEdge = pd.read_pickle(DATAPATH + df_list[0])
for i in range(1, len(df_list)):
    df_SPARC_Abhi_ONEdge = df_SPARC_Abhi_ONEdge.join(pd.read_pickle(DATAPATH+df_list[i]))
df_SPARC_Abhi_ONEdge.index.name = 'time'
df_SPARC_Abhi_ONEdge = df_SPARC_Abhi_ONEdge.reset_index()

#plot data and save figure
f, ax = plt.subplots(1,5, sharey=True, figsize=(15,5))
[ax[i].plot(df_SPARC_Abhi_ONEdge['Mean_PD_'+str(velocity[i])], 'b', label='PD_SPARC_Gcamp') for i in range(5)];
[ax[i].plot(df_SPARC_Abhi_ONEdge['Mean_ND_'+str(velocity[i])], 'r', label='ND_SPARC_Gcamp') for i in range(5)];
plt.legend();
plt.savefig('figures/Abhishek_SPARC_GCamp_ONEdge.pdf', dpi = 1000)

df_arclight = Bulle_arclight_ON_Edge_data.copy() #arclight data copy
dt = df_arclight.iloc[7]['time'] - df_arclight.iloc[6]['time'] #time step dt

#stimulus used for error calculation
velocity = [15, 30, 60,120,240]; stim=[]
[stim.extend(['Mean_PD_'+str(velocity[i]),'Mean_ND_'+str(velocity[i])]) for i in range(len(velocity))];
#[stim.extend(['Mean_PD_'+str(velocity[i])]) for i in range(len(velocity))];
print (f'Stimulus used for error calculation {stim}')

#may use less number of parameter values to search through
thres = np.linspace(-0.04, 0.10, 20) #threshold value search
tau =np.linspace(0.05,1.5,20) #time constants
gain = np.linspace(50.0, 200.0, 20) #gain
error = np.zeros((len(thres), len(tau), len(gain))) #array to store error values
#calculating error going through for loops for different parameter values and stimulus condtions
for i in range(len(thres)):
    for j in range(len(tau)):
        for k in range(len(gain)):
            err_= 0
            for sti in stim:
                PD_Gcamp_data = np.array(df_SPARC_Abhi_ONEdge[sti])
                PD_arclight = np.array(df_arclight[sti])
                model_data = model.Ca_model(PD_arclight.copy(), tau[j], dt, thres[i], gain[k])
                err_ += model.peak_error_calc(PD_Gcamp_data.copy(), model_data.copy())
            error[i, j, k] = err_/len(stim)
print(f'Minimum error {error.min()}')

#parameters corresponding to minimum error
thres_model = thres[np.argwhere(error == np.min(error))[0][0]]
tau_model = tau[np.argwhere(error == np.min(error))[0][1]]
gain_model = gain[np.argwhere(error == np.min(error))[0][2]]
print("Parameters values corresponding to minimum error : ")
print(f'threshold : {thres_model},time constant : {tau_model}, gain:{gain_model}')

#print final figure using model parameters corresponding to minimum error
velocity = [15, 30, 60, 120, 240]
f, ax = plt.subplots(1,5, sharey=True, figsize=(15,5))
[ax[i].plot(df_SPARC_Abhi_ONEdge['Mean_PD_'+str(velocity[i])],'b',label='SPARCGcamp_PD_expt_data') for i in range(5)];
[ax[i].plot(model.Ca_model(df_arclight['Mean_PD_'+str(velocity[i])].copy(),
                    tau_model, dt, thres_model, gain_model), 'b--', label='Gcamp_PD_model', alpha=0.8) for i in range(5)];

[ax[i].plot(df_SPARC_Abhi_ONEdge['Mean_ND_'+str(velocity[i])],'r',label='SPARCGcamp_ND_expt_data') for i in range(5)];
[ax[i].plot(model.Ca_model(df_arclight['Mean_ND_'+str(velocity[i])].copy(),
                    tau_model, dt, thres_model, gain_model), 'r--', label='Gcamp_ND_model', alpha=0.8) for i in range(5)];
plt.legend();
plt.savefig('figures/SPARC_ONEdge_Model.pdf', dpi = 1000)









