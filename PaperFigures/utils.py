#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:18:05 2021

@author: amishra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.ticker import FormatStrFormatter
from mpl_axes_aligner import align

def plot_data(data_list, dt=0.0769, fig_size=(10,5), speed=[15,30,60,120], title='', savefig=False, c=['k','b']):
    """Plots data for multiple datsets"""
    n_col = len(speed) #number of columns i.e. number of speed stimuli
    n_sti = data_list[0].shape[1] #number of orientations
    #x_values = np.arange(data_list[0].shape[0]) * dt #changing x-axis to time. multiply with time step dt = 1/frequency
    f, ax = plt.subplots(int(n_sti/n_col), int(n_col), sharex=True, sharey= True, figsize=fig_size)
    c_count = 0 #color count for different dataset
    for data in data_list:
        count = 0 #count for stimuli
        for i in range(int(n_sti/n_col)):
            for j in range(n_col):
                #ax[i, j].plot(x_values, data[:, count], color=c[c_count],linewidth=2.0)
                ax[i, j].plot(data[:, count], color=c[c_count], linewidth=2.0)
                count = count + 1
        c_count += 1
    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    if savefig:
        plt.savefig('figures/'+title+'.pdf',dpi=1000);
        
def plot_data_twinaxis(data_list, dt=1/13.0, fig_size=(10,5), speed=[15,30,60,120], title='', savefig=False, c=['k','r'],ylim1=(-0.04,0.08),ylim2=(-20,250),contrast=False):
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
                    ax[i, j].plot(x_values, data[:, count], label='Arclight',color=c[c_count],linewidth=2.0)
                    ax[i,j].set_ylim(ylim1)
                    #ax[i, j].set_yticks(-0.04,0.08);
                    #ax[i,j].yaxis.set_ticks(np.arange(-1.0, 2.0, 0.02))
                    if contrast == True:
                        if count == 7:
                            ax[i,j].legend(loc=1,frameon=False);
                    else:
                        if count == 3:
                            ax[i,j].legend(loc=1,frameon=False);
                        
                    
                else :
                    ax1 = ax[i,j].twinx()
                    ax1.plot(x_values, data[:, count], label='GCaMP',color=c[c_count],linewidth=2.0)
                    ax1.set_ylim(ylim2)
                    ax1.spines['right'].set_color('red')
                    ax1.tick_params(axis='y', colors='red')
                    if (count != 3) and (count != 7):
                        ax1.set_yticklabels([])
                    if contrast == True :
                        if count == 7:
                            ax1.legend(loc=1,bbox_to_anchor=(0,0,1,0.9), frameon=False);
                    else:
                        if count == 3:
                            ax1.legend(loc=1,bbox_to_anchor=(0,0,1,0.9), frameon=False);
                        
                    #Adjust plotting range of two y axes
                    org1 = 0.0
                    org2 = 0.0
                    pos = 0.2
                    align.yaxes(ax[i,j], org1, ax1, org2, pos)
                    #ax1.set_yticks(ylim,0.2)
                #ax[i, j].plot(data[:, count], color=c[c_count])
                
                count = count + 1
        c_count += 1
    
    
    row = ['PD','ND']
    for axis,s in zip(ax[0],speed):
        if contrast == True:
            axis.set_title(str(s)+'%',fontsize=12);
        else:
            axis.set_title(str(s)+' deg/s',fontsize=12);
    for axis,r in zip(ax[:,0], row):
        axis.set_ylabel(r, rotation=0,fontsize=12, labelpad=10)
    ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xticks(np.arange(0,10.0,2.0))
    #if contrast == True :
    #    f.text(0.01,0.95,'Contrast(%)',fontsize=10)
    #else:
    #    f.text(0.01,0.95,'Speed(deg/sec.)',fontsize=10)
        
    
    f.text(0.5, 0.0, r'Time(seconds)', fontsize=11, ha='center')
    f.text(0.01, 0.5, r'Voltage response $(-\Delta F/F)$', fontsize=10, va='center', rotation='vertical')
    f.text(0.97, 0.5, r'Calcium response $(\Delta F/F)$', color='red',fontsize=10, va='center', rotation='vertical')
    #f.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
    #ax.set_xlabel('Time(seconds')
    #plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(right=0.92)
    if savefig:
        plt.savefig('figures/'+title+'.pdf',dpi=1000);
        
def plot_peaktuning(data_list, speed, color, ylim1=(0.0,0.12),ylim2=(0.0,6.0), fig_size=(3,5.5), title='',savefig=False, contrast=False):
    peak_tuning = np.empty((len(data_list),len(speed)*2)) #first dimesnion is for Arclight, Gcamp, Model respectively. 
                                #second dimension is for n speeds PD first n column, ND last n columns
    n=len(speed)
    for i in range(len(data_list)):
        peak_tuning[i,:] = data_list[i].max(axis=0)
    f, ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=fig_size)   
    for i in range(peak_tuning.shape[0]):
        ax[0].plot(peak_tuning[0][:n],marker='o',color=color[0],label='Arclight');
        ax[0].set_ylim(ylim1)
        ax[0].set_ylabel('PD', rotation=0,fontsize=12,labelpad=10)
        ax1 = ax[0].twinx()
        ax1.plot(peak_tuning[1][:n],marker='o',color=color[1],label='GCaMP');
        #ax1.plot(peak_tuning[2][:n],marker='o',color=color[2]);
        ax1.set_ylim(ylim2)
        ax1.spines['right'].set_color('red')
        ax1.tick_params(axis='y', colors='red')
        
        ax[1].plot(peak_tuning[0][n:],marker='o',color=color[0],label='Arclight');
        ax[1].set_ylim(ylim1)
        ax[1].set_ylabel('ND', rotation=0,fontsize=12, labelpad=10)
        ax2 = ax[1].twinx()
        ax2.plot(peak_tuning[1][n:],marker='o',color=color[1],label='GCaMP');
        #ax2.plot(peak_tuning[2][n:],marker='o',color=color[2]);
        ax2.set_ylim(ylim2);
        if i == 0:
            ax[1].legend(loc=1, frameon=False);
        ax2.legend(loc=1, bbox_to_anchor=(0.0,0,1,0.9),frameon=False);
        ax2.spines['right'].set_color('red');
        ax2.tick_params(axis='y', colors='red');
    #ax[0].legend(loc=0,frameon=False);
    ax[0].set_title(r'Peak $\Delta F/F$')
    if contrast:
        ax[1].set_xlabel('Contrast(%)',fontsize=12)
    else:
        ax[1].set_xlabel('Speed(deg/s)',fontsize=12)
        
    #ax[0].xaxis.set_ticks([0.0,1.0,2.0,3.0,4.0],velocity);
    #f.subplots_adjust(hspace=0.2)
    plt.xticks(range(0,n),speed); 
    plt.tight_layout();
    if savefig:
        plt.savefig('figures/'+title+'.pdf',dpi=1000);
        
        
def load_data(filename):
    x = np.genfromtxt(filename, delimiter = '\t')
    return x   

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

def Ca_model(x, thres, tauhp, taulp1, gain1, taulp2, gain2, dt=0.0769):
    x_thres = threshold_cut(x,thres)
    x_thres_bp1 = bandpass(x_thres, tauhp, taulp1, dt)
    x_1 = x_thres_bp1 * gain1
    x_thres_bp2 = bandpass(x_thres, tauhp, taulp2, dt)
    x_2 = x_thres_bp2 * gain2
    y = x_1 + x_2
    #y = np.roll(y, int(tshift), axis=0)
    return y

def Ca_model_1(x, thres, tauhp, taulp1, gain1, dt=0.0769):
    x_thres = threshold_cut(x,thres)
    x_thres_hp = highpass(x_thres, tauhp, dt)
    x_thres_hp_lp = lowpass(x_thres_hp, taulp1, dt)
    y = x_thres_hp_lp * gain1
    return y

def Ca_model_2(x, thres, tauhp, taulp1, taulp2, gain, dt=0.0769):
    x_thres = threshold_cut(x, thres)
    x_thres_hp = highpass(x_thres, tauhp, dt)
    x_thres_hp_lp1 = lowpass(x_thres_hp, taulp1, dt)
    #x_thres_hp_lp1 = shift_signal(x_thres_hp_lp1, T4Ca_model, vel=15.0)
    x_thres_hp_lp2 = lowpass(x_thres_hp, taulp2, dt) 
    #x_thres_hp_lp2 = shift_signal(x_thres_hp_lp2, T4Ca_model, vel=15.0)
    y = (x_thres_hp_lp1 + x_thres_hp_lp2)*gain
    #plot_data([x, x_thres, x_thres_hp, x_thres_hp_lp1, x_thres_hp_lp2, y], c=['k','r','g','blue','brown','grey']);
    return y

def Ca_model_3(x, thres, tauhp, taulp1, taulp2, gain1, gain2, dt=0.0769):
    x_thres = threshold_cut(x, thres)
    x_thres_hp = highpass(x_thres, tauhp, dt)
    x_thres_hp_lp1 = lowpass(x_thres_hp, taulp1, dt)
    x_thres_hp_lp1 = x_thres_hp_lp1 * gain1
    #x_thres_hp_lp1 = shift_signal(x_thres_hp_lp1, T4Ca_model, vel=15.0)
    x_thres_hp_lp2 = lowpass(x_thres_hp, taulp2, dt)
    x_thres_hp_lp2 = x_thres_hp_lp2 * gain2
    #x_thres_hp_lp2 = shift_signal(x_thres_hp_lp2, T4Ca_model, vel=15.0)
    y = x_thres_hp_lp1 + x_thres_hp_lp2
    #plot_data([x, x_thres, x_thres_hp, x_thres_hp_lp1, x_thres_hp_lp2, y], c=['k','r','g','blue','brown','grey']);
    return y

def Ca_model_4(x, thres, taulp1, taulp2, gain1, gain2, dt=0.0769, plot=False):
    x_thres = threshold_cut(x.copy(), thres)
    x_thres_lp2 = lowpass(x_thres.copy(), taulp2, dt)
    x_thres_lp2 = x_thres_lp2 * gain2
    x_thres_lp1 = lowpass(x_thres.copy(), taulp1, dt)
    x_thres_lp1 = x_thres_lp1 * gain1
    x_thres_lp1 = x_thres_lp1 * x_thres_lp2
    y = x_thres_lp1 + x_thres_lp2
    #y=x_thres_lp2
    if plot==True:
        plot_data([x, x_thres, x_thres_lp1, x_thres_lp2, y], c=['k','r','g','brown','blue']);
    return y



def shift_signal(gcamp_data, model_data, vel):
    if vel == 15.0:
        shift_columns = np.argmax(gcamp_data[40:60,:],axis=0)-np.argmax(model_data[40:60,:],axis=0)
    elif vel == 30.0:
        shift_columns = np.argmax(gcamp_data[10:20,:],axis=0)-np.argmax(model_data[10:20,:],axis=0)
    elif vel == 60.0:
        shift_columns = np.argmax(gcamp_data[10:20,:],axis=0)-np.argmax(model_data[10:20,:],axis=0)
    A = model_data
    r = shift_columns
    rows, columns = np.ogrid[:A.shape[0], :A.shape[1]]
    r[r < 0] += A.shape[0]
    rows = rows - r[np.newaxis,:]
    model_shift = A[rows, columns]
    return model_shift

def shift_signal_edge(gcamp_data, model_data):
    shift_columns = np.argmax(gcamp_data,axis=0)-np.argmax(model_data,axis=0)
    A = model_data
    r = shift_columns
    rows, columns = np.ogrid[:A.shape[0], :A.shape[1]]
    r[r < 0] += A.shape[0]
    rows = rows - r[np.newaxis,:]
    model_shift = A[rows, columns]
    return model_shift

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        