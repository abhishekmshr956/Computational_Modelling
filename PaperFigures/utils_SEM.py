#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:02:31 2021

@author: amishra
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.ticker import FormatStrFormatter
from mpl_axes_aligner import align

def plot_dataSEM_twinaxis(data_list, SEM_list, dt=1/13.0, fig_size=(10,5), speed=[15,30,60,120], title='', savefig=False, c=['k','r'],ylim1=(-0.04,0.08),ylim2=(-20,250),contrast=False):
    """Plots data for multiple datsets"""
    n_col = len(speed) #number of columns i.e. number of speed stimuli
    n_sti = data_list[0].shape[1] #number of orientations
    x_values = np.arange(data_list[0].shape[0]) * dt #changing x-axis to time. multiply with time step dt = 1/frequency
    f, ax = plt.subplots(int(n_sti/n_col), int(n_col), sharex=True, sharey= True, figsize=fig_size)
    c_count = 0 #color count for different dataset
    for data, SEM in zip(data_list, SEM_list):
        count = 0 #count for stimuli
        for i in range(int(n_sti/n_col)):
            for j in range(n_col):
                if c_count == 0:
                    ax[i, j].plot(x_values, data[:, count], label='Arclight',color=c[c_count],linewidth=2.0)
                    ax[i, j].fill_between(x_values, data[:, count]-SEM[:, count], data[:, count]+SEM[:, count],color=c[c_count],alpha=0.3)
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
                    ax1.fill_between(x_values, data[:, count]-SEM[:, count], data[:, count]+SEM[:, count],color=c[c_count],alpha=0.3)
                    
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
        
    
    f.text(0.5, 0.007, r'Time(seconds)', fontsize=11, ha='center')
    f.text(0.01, 0.5, r'Voltage response $(-\Delta F/F)$', fontsize=10, va='center', rotation='vertical')
    f.text(0.97, 0.5, r'Calcium response $(\Delta F/F)$', color='red',fontsize=10, va='center', rotation='vertical')
    #f.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
    #ax.set_xlabel('Time(seconds')
    #plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    #plt.subplots_adjust(right=0.92)
    if savefig:
        plt.savefig('figures/'+title+'.pdf',dpi=1000);
        
        
def plot_twinaxis(data_list, SEM_list, dt=1/13.0, fig_size=(10,5), speed=[15,30,60,120], title='', savefig=False, c=['k','r'],ylim1=(-0.04,0.08),ylim2=(-20,250),contrast=False):
    """Plots data for multiple datsets"""
    n_col = len(speed) #number of columns i.e. number of speed stimuli
    n_sti = data_list[0].shape[1] #number of orientations
    x_values = np.arange(data_list[0].shape[0]) * dt #changing x-axis to time. multiply with time step dt = 1/frequency
    f, ax = plt.subplots(2,7, sharex='col', sharey= True, figsize=fig_size,constrained_layout=True)
    c_count = 0 #color count for different dataset
    for data, SEM in zip(data_list, SEM_list):
        count = 0 #count for stimuli
        for i in range(int(n_sti/n_col)):
            for j in range(n_col):
                if c_count == 0:
                    ax[i, j].plot(x_values, data[:, count], label='Arclight',color=c[c_count],linewidth=2.0)
                    ax[i, j].fill_between(x_values, data[:, count]-SEM[:, count], data[:, count]+SEM[:, count],color=c[c_count],alpha=0.3)
                    ax[i,j].set_ylim(ylim1)
                    ax[i,j].yaxis.set_ticks(np.arange(-0.04,ylim1[1]+0.08,0.04));
                    ax[i,j].set_xticks(np.arange(0,10.0,2.0))
                    #ax[i, j].set_yticks(np.arange(-0.04,0.12,0.02));
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
                    ax1.fill_between(x_values, data[:, count]-SEM[:, count], data[:, count]+SEM[:, count],color=c[c_count],alpha=0.3)
                    
                    ax1.set_ylim(ylim2)
                    #ax1.yaxis.set_ticks(np.arange(-1.0,ylim2[1]+0.5,1.0));
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
                    pos = 0.20
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
    ax[1,1].set_xlabel('Time(seconds)',fontsize=12);
    ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    #if contrast == True :
    #    f.text(0.01,0.95,'Contrast(%)',fontsize=10)
    #else:
    #    f.text(0.01,0.95,'Speed(deg/sec.)',fontsize=10)
    ax[0,4].axis('off')
    ax[1,4].axis('off')
    #ax[0,6].axis('off')
    ax[1,6].axis('off')
    #ax[1,7].axis('off')
        
    
    #f.text(0.3, 0.0, r'Time(seconds)', fontsize=11, ha='center')
    f.text(0.005, 0.55, r'Voltage response $(-\Delta F/F)$', fontsize=11, va='center', rotation='vertical')
    f.text(0.57, 0.55, r'Calcium response $(\Delta F/F)$', color='red',fontsize=11, va='center', rotation='vertical')
    #f.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
    #ax.set_xlabel('Time(seconds')
    #plt.suptitle(title, fontsize=15)
    #plt.tight_layout()
    #plt.subplots_adjust(right=0.92)
    axp = ax[0,5]
            
    axp.get_shared_y_axes().remove(axp)
    
        # Create and assign new ticker
    yticker = matplotlib.axis.Ticker()
    axp.yaxis.major = yticker
    
    # The new ticker needs new locator and formatters
    yloc = matplotlib.ticker.AutoLocator()
    yfmt = matplotlib.ticker.ScalarFormatter()
    
    axp.yaxis.set_major_locator(yloc)
    axp.yaxis.set_major_formatter(yfmt)
    
    axp = ax[1,5]
            
    axp.get_shared_y_axes().remove(axp)
    
        # Create and assign new ticker
    yticker = matplotlib.axis.Ticker()
    axp.yaxis.major = yticker
    
    # The new ticker needs new locator and formatters
    yloc = matplotlib.ticker.AutoLocator()
    yfmt = matplotlib.ticker.ScalarFormatter()
    
    axp.yaxis.set_major_locator(yloc)
    axp.yaxis.set_major_formatter(yfmt)
    
    
    #peaktuning plot
    peak_tuning = np.empty((len(data_list),len(speed)*2)) #first dimesnion is for Arclight, Gcamp, Model respectively. 
                                #second dimension is for n speeds PD first n column, ND last n columns
    n=len(speed)
    for i in range(len(data_list)):
        peak_tuning[i,:] = data_list[i].max(axis=0)
    for i in range(peak_tuning.shape[0]):
        ax[0,5].plot(peak_tuning[0][:n],marker='o',color=c[0],label='Arclight');
        ax[0,5].set_ylim(0,ylim1[1])
        ax[0,5].yaxis.set_ticks(np.arange(0,ylim1[1]+0.08,0.04));
        ax[0,5].set_ylabel('PD', rotation=0,fontsize=12,labelpad=10)
        ax[0,5].yaxis.set_tick_params(labelbottom=True)
        #ax[0,5].set_xticks(range(0,n),speed);
        ax1 = ax[0,5].twinx()
        ax1.plot(peak_tuning[1][:n],marker='o',color=c[1],label='GCaMP');
        #ax1.plot(peak_tuning[2][:n],marker='o',color=color[2]);
        ax1.set_ylim(0,ylim2[1])
        ax1.spines['right'].set_color('red')
        ax1.tick_params(axis='y', colors='red')
        
        ax[1,5].plot(peak_tuning[0][n:],marker='o',color=c[0],label='Arclight');
        ax[1,5].set_ylim(0,ylim1[1])
        ax[1,5].yaxis.set_ticks(np.arange(0,ylim1[1]+0.04,0.04));
        ax[1,5].set_ylabel('ND', rotation=0,fontsize=12, labelpad=10)
        ax[1,5].yaxis.set_tick_params(labelbottom=True)
        ax2 = ax[1,5].twinx()
        ax2.plot(peak_tuning[1][n:],marker='o',color=c[1],label='GCaMP');
        #ax2.plot(peak_tuning[2][n:],marker='o',color=color[2]);
        ax2.set_ylim(0,ylim2[1]);
        if i == 0:
            ax[1,5].legend(loc=1, frameon=False);
        ax2.legend(loc=1, bbox_to_anchor=(0.0,0,1,0.9),frameon=False);
        ax2.spines['right'].set_color('red');
        ax2.tick_params(axis='y', colors='red');
        #ax[1,5].set_xticks(range(0,n),speed);
        plt.xticks(range(0,n),speed); 
        
        #Adjust plotting range of two y axes
        # org1 = 0.0
        # org2 = 0.0
        # pos = 0.05
        # align.yaxes(ax[0,5], org1, ax1, org2, pos)
        # align.yaxes(ax[1,5], org1, ax2, org2, pos)
        
    ax[0,5].set_title(r'Peak $\Delta F/F$',fontsize=12)
    #ax[1,5].set_xlabel('Speed(deg/s)');
    if contrast:
        #f.text(0.6, 0.007, 'Contrast(%)', fontsize=11, ha='center')
        ax[1,5].set_xlabel('Contrast(%)',fontsize=12)
    else:
        ax[1,5].set_xlabel('Speed(deg/s)',fontsize=12)
        #f.text(0.65, 0.007, 'Speed(deg/s)', fontsize=11, ha='center')
        
        
    ### DSI plot
    dsi = np.empty((len(data_list), len(speed)))
    for i in range(peak_tuning.shape[0]):
        for j in range(n):
            PD = peak_tuning[i,j]
            ND = peak_tuning[i,j+4]
            dsi[i,j] = (PD-ND) / (PD+ND)
    axdsi = ax[0,6]
            
    axdsi.get_shared_y_axes().remove(axdsi)
    
        # Create and assign new ticker
    yticker = matplotlib.axis.Ticker()
    axdsi.yaxis.major = yticker
    
    # The new ticker needs new locator and formatters
    yloc = matplotlib.ticker.AutoLocator()
    yfmt = matplotlib.ticker.ScalarFormatter()
    
    axdsi.yaxis.set_major_locator(yloc)
    axdsi.yaxis.set_major_formatter(yfmt)
    #shay = ax[0,7].get_shared_y_axes()
    #shay.remove(ax[0,7])
    #ax[0,7].plot(peak_tuning[1][:],marker='o',color=c[0],label='Arclight');
    axdsi.plot(dsi[0],marker='o',color=c[0],label='Arclight');
    axdsi.plot(dsi[1],marker='o',color=c[1],label='GCaMP');
    axdsi.set_ylim(0,1.1);
    axdsi.legend(loc=0,bbox_to_anchor=(0.0,0,1,0.7),frameon=False);
    axdsi.yaxis.set_ticks(np.arange(0,1.2,0.2));
    axdsi.xaxis.set_tick_params(labelbottom=True)
    axdsi.yaxis.set_tick_params(labelbottom=True)
    axdsi.xaxis.set_ticks(range(0,n)); 
    axdsi.set_xticklabels([15,30,60,120]);
    axdsi.set_title('DSI',fontsize=12);
    axdsi.set_xlabel('Speed(deg/s)',fontsize=12);
    #plt.setp(ax[0,7].get_xticklabels(), visible=True)
    #plt.tight_layout();
    #f.tight_layout(pad=0.1);
    #plt.subplots_adjust(bottom=0.1)
    
    
    
    

        
    

    
    if savefig:
        plt.savefig('figures/'+title+'.pdf',dpi=1000);
        
        
        
        
        
        
        
        
        