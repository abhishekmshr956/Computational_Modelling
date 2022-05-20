#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:56:17 2022

@author: amishra
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from mpl_axes_aligner import align

rotations = np.arange(0.0,390.0,30.0)
rotations_rad = (rotations * np.pi) / 180

def cal_peaktuning(data_list, ncol=8):
    peak_tuning = np.empty((len(data_list), ncol)) #first dimesnion is for Arclight, Gcamp, Model respectively. 
                                #second dimension is for 4 speeds PD first 4 column, ND last 4 columns
    for i in range(len(data_list)):
        peak_tuning[i,:] = data_list[i].max(axis=0)
        
    return peak_tuning

def cal_peaktuning_sem(data_list, sem_list, ncol=8):
    peak_tuning = np.empty((len(data_list), ncol)) #first dimesnion is for Arclight, Gcamp, Model respectively. 
                                #second dimension is for 4 speeds PD first 4 column, ND last 4 columns
    sem_tuning = np.empty((len(data_list), ncol)) #first dimesnion is for Arclight, Gcamp, Model respectively. 
                                #second dimension is for 4 speeds PD first 4 column, ND last 4 columns
    for i in range(len(data_list)):
        peak_tuning[i,:] = data_list[i].max(axis=0)
        arglist = data_list[i].argmax(axis=0)
        sem_tuning[i,:] = [sem_list[i][arglist[j],j] for j in range(len(arglist))]
        
    return peak_tuning, sem_tuning                      

def cal_dsi(data_list):
    peak_tuning = cal_peaktuning(data_list)
    dsi = np.empty((len(data_list),4))
    for i in range(peak_tuning.shape[0]):
        for j in range(4):
            PD = peak_tuning[i,j]
            ND = peak_tuning[i,j+4]
            dsi[i,j] = (PD-ND) / (PD+ND)
            
    return dsi

def cal_dsi_sem(data_list, sem_list):
    peak_tuning, sem_tuning = cal_peaktuning_sem(data_list, sem_list)
    dsi = np.empty((len(data_list),4))
    dsi_sem = np.empty((len(data_list),4))
    for i in range(peak_tuning.shape[0]):
        for j in range(4):
            PD = peak_tuning[i,j]
            PD_sem = sem_tuning[i,j]
            ND = peak_tuning[i,j+4]
            ND_sem = sem_tuning[i,j+4]
            dsi[i,j] = (PD-ND) / (PD+ND)
            dsi_sem[i,j] = (PD_sem-ND_sem) / (PD_sem+ND_sem)
            
    return dsi, dsi_sem

def setmyaxes(myxpos, myypos, myxsize, myysize, myylim):
    
    ax = plt.axes([myxpos, myypos, myxsize, myysize])
    ax.set_ylim(myylim)
    #ax.set_xticks(myxticks)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
def setmyaxes_polar(myxpos, myypos, myxsize, myysize, myylim):
    
    ax = plt.axes([myxpos, myypos, myxsize, myysize], polar=True)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')


def plot_polars(peak_tuning, peak_tuning_sem, upperm, mytitle):
    xsize=0.20
    ysize=0.15
    xoffs=0.24
    leftm=0.05
    
    for i in range(4):
        setmyaxes_polar(i*xoffs+leftm,upperm-0.1,xsize,ysize,(0.0,1.2))
        data = peak_tuning[0,i,:]
        data_sem = peak_tuning_sem[0,i,:]
        data = np.append(data, data[0])
        data_sem = np.append(data_sem, data_sem[0])
        ax=plt.gca()
        ax.plot(rotations_rad, data, marker='o', markersize=2.0,color='k',label='Arclight',linewidth=1.0)
        ax.fill_between(rotations_rad, data-data_sem, data+data_sem, color='k',alpha=0.2)
        ax.set_title(mytitle[i], fontsize=10)
        ax.set_ylim((0.0,1.2))
        if i==1: ax.legend(loc=1, bbox_to_anchor=(0,0,1.21,1.18),frameon=False, prop={'size':7});
        #plt.yticks(fontsize=6)
        
        data = peak_tuning[1,i,:]
        data_sem = peak_tuning_sem[1,i,:]
        data = np.append(data, data[1])
        data_sem = np.append(data_sem, data_sem[1])
        ax.plot(rotations_rad, data, marker='o', markersize=2.0,color='r',label='GCaMP',linewidth=1.0)
        ax.fill_between(rotations_rad, data-data_sem, data+data_sem, color='r',alpha=0.2)
        if i==1: ax.legend(loc=1, bbox_to_anchor=(0,0,1.21,1.18),frameon=False, prop={'size':7});
        
        ax.set_rticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['','','','','',1.0])
        ax.set_rlabel_position(22.5)
        ax.xaxis.set_tick_params(pad=-5)
        #ax.set_thetagrids(rotations[:-1],frac=0.5)
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)
        
        
        
    #plt.text(300,1000*upperm+110,mytitle,fontsize=12,transform=None)
    
def plot_Ldir(data_list, xlabel,c,legend,xticklabel,yoffs=0.10):
    xsize=0.20
    ysize=0.15
    xoffs=0.35
    yoffs=yoffs
    leftm=0.20
    
    mytitle=['Grating Speed', 'Grating Contrast']
    xlabel=['speed [deg/s]','contrast[%]']
    
    for i in range(2):
        setmyaxes(i*xoffs+leftm,yoffs,xsize,ysize,(0.0,1.0))
        #Ldir_mag = Ldir_mag_list[i]
        ax=plt.gca()
        plot_data_witherrorbar(data_list[2*i][:,0], data_list[2*i][:,1], c[0], legend[0])
        plot_data_witherrorbar(data_list[2*i+1][:,0], data_list[2*i+1][:,1], c[1], legend[1])
        #ax.plot(Ldir_mag[0],marker='o',markersize=6,color=c[0],label=legend[0]);
        #ax.plot(Ldir_mag[1],marker='o',markersize=6,color=c[1],label=legend[1]);
        ax=plt.gca()
        # get handles
        handles, labels = ax.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        plt.legend(handles,labels,loc=0,frameon=False, prop={'size':8})
        
        #ax.legend(loc=0,frameon=False, prop={'size':7});
        plt.ylabel('Ldir-index')
        plt.xlabel(xlabel[i])
        plt.title(mytitle[i],fontsize=12)
        plt.xticks(range(0,4),xticklabel[i])
    
def plot_timedata(x, y, sem, c, l):
    
    plt.plot(x,y, color=c, label=l)
    ax = plt.gca()
    ax.fill_between(x,y-sem,y+sem, color=c, alpha=0.5)
    
def plot_data(data,c,l):
    plt.plot(data,marker='o',markersize=6,color=c,label=l)
    
def plot_data_witherrorbar(data, sem,c,l):
    #plt.plot(data,marker='o',markersize=6,color=c, label=l)
    plt.errorbar(range(4), data,yerr=sem,marker='o',markersize=3,color=c, label=l)
    # ax = plt.gca()
    # ax.fill_between(range(4),data-sem,data+sem, color=c, alpha=0.2)
    
def plot_newsignals(data_list, sem_list, data_list1, myylim, upperm, mytitle, myxlabel,xticklabel,c):
    
    xsize = 0.15
    ysize = 0.10
    xoffs = 0.167
    yoffs = 0.15
    leftm = 0.08
    
    #time-course of signals
    
    dt = 1/13.0
    x_values = np.arange(data_list[0].shape[0])*dt #changing x-axis to time. multiply with time step dt = 1/frequency
    count = 0 #count for stimuli
    
    for j in range(2):
        for i in range(4):
            setmyaxes(i*xoffs+leftm, upperm-j*yoffs,xsize,ysize,myylim[0])
            plt.xticks(np.arange(0,10.0,4.0))
            ydata = data_list[0][:,count]
            semdata = sem_list[0][:,count]
            plot_timedata(x_values, ydata, semdata, c[0], 'Arclight')
            plt.margins(0)
            #plt.yticks(np.arange(myylim[0][0],myylim[0][1],0.05))
            ax = plt.gca()
            ax.set_ylim(myylim[0])
            ax.set_yticks(np.arange(myylim[0][0],myylim[0][1]-myylim[0][0],-myylim[0][0]))
            if i!=0: ax.set_yticklabels([])
            if j==0: ax.set_title(mytitle[i], fontsize=10)
            if j==1 and i==3: ax.legend(loc=1,frameon=False, prop={'size':8})
            if j==0 and i==0: ax.set_ylabel('PD', rotation=0,fontsize=10, labelpad=3)
            if j==1 and i==0: ax.set_ylabel('ND',rotation=0,fontsize=10, labelpad=3)
            
            ax2 = ax.twinx()
            ax2.set_ylim(myylim[1])
            ax2.set_yticks(np.arange(myylim[1][0],myylim[1][1]-myylim[1][0],-myylim[1][0]))
            plt.sca(ax2)
            plot_timedata(x_values, data_list[1][:,count], sem_list[1][:,count],c[1], 'GCaMP' )
            #Adjust plotting range of two y axes
            org1 = 0.0
            org2 = 0.0
            pos = 0.25
            align.yaxes(ax, org1, ax2, org2, pos)
            plt.margins(0)
            ax2.spines['right'].set_color('red')
            ax2.tick_params(axis='y', colors='red')
            ax2.tick_params(pad=2)
            if i!=3: ax2.set_yticklabels([])
            if j==1 and i==3: ax2.legend(loc=1,bbox_to_anchor=(0,0,1,0.85),frameon=False, prop={'size':8})
            count = count + 1
            
           
    # peak dF/F
    #(peak_tuning, sem_tuning) = cal_peaktuning_sem(data_list, sem_list)
    for j in range(2):
        setmyaxes(4.47*xoffs+leftm, upperm-j*yoffs,xsize,ysize,(0,myylim[0][1]))
        #plot_data(peak_tuning[0][j*4:4+j*4],'k','Arclight')
        #plot_data_witherrorbar(peak_tuning[0][j*4:4+j*4], sem_tuning[0][j*4:4+j*4], 'k', 'Arclight')
        plot_data_witherrorbar(data_list1[0][:,2*j], data_list1[0][:,2*j+1], 'k', 'Arclight')
        if j==0: plt.title(r'Peak $\Delta$F/F')
        if j==1: plt.xlabel(myxlabel)
        if j==1: 
            ax=plt.gca()
            # get handles
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            plt.legend(handles, labels,loc=1,frameon=False, prop={'size':8})
        ax = plt.gca()
        ax2= ax.twinx()
        ax2.set_ylim((0,myylim[1][1]))
        plt.sca(ax2)
        #plot_data(peak_tuning[1][j*4:4+j*4],'r','GCaMP')
        plot_data_witherrorbar(data_list1[1][:,2*j], data_list1[1][:,2*j+1], 'r', 'GCaMP')
        
        ax2.spines['right'].set_color('red')
        ax2.tick_params(axis='y', colors='red')
        if j==1: 
            ax=plt.gca()
            # get handles
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            plt.legend(handles,labels,loc=1,bbox_to_anchor=(0,0,1,0.85),frameon=False, prop={'size':8})
        
        plt.xticks(range(0,4),xticklabel)
        
            
    #plt.text(220,1000*upperm+110,'Grating Speed',fontsize=12,transform=None)
    #plt.text(200,600*upperm+100,mytitle,fontsize=12,transform=None)

        
def plot_signals(data_list, sem_list, myylim, upperm, mytitle, myxlabel,xticklabel,c):
    
    xsize = 0.15
    ysize = 0.10
    xoffs = 0.167
    yoffs = 0.15
    leftm = 0.08
    
    #time-course of signals
    
    dt = 1/13.0
    x_values = np.arange(data_list[0].shape[0])*dt #changing x-axis to time. multiply with time step dt = 1/frequency
    count = 0 #count for stimuli
    
    for j in range(2):
        for i in range(4):
            setmyaxes(i*xoffs+leftm, upperm-j*yoffs,xsize,ysize,myylim[0])
            plt.xticks(np.arange(0,10.0,4.0))
            ydata = data_list[0][:,count]
            semdata = sem_list[0][:,count]
            plot_timedata(x_values, ydata, semdata, c[0], 'Arclight')
            plt.margins(0)
            #plt.yticks(np.arange(myylim[0][0],myylim[0][1],0.05))
            ax = plt.gca()
            ax.set_ylim(myylim[0])
            ax.set_yticks(np.arange(myylim[0][0],myylim[0][1]-myylim[0][0],-myylim[0][0]))
            if i!=0: ax.set_yticklabels([])
            if j==0: ax.set_title(mytitle[i], fontsize=10)
            if j==1 and i==3: ax.legend(loc=1,frameon=False, prop={'size':8})
            if j==0 and i==0: ax.set_ylabel('PD', rotation=0,fontsize=10, labelpad=3)
            if j==1 and i==0: ax.set_ylabel('ND',rotation=0,fontsize=10, labelpad=3)
            
            ax2 = ax.twinx()
            ax2.set_ylim(myylim[1])
            ax2.set_yticks(np.arange(myylim[1][0],myylim[1][1]-myylim[1][0],-myylim[1][0]))
            plt.sca(ax2)
            plot_timedata(x_values, data_list[1][:,count], sem_list[1][:,count],c[1], 'GCaMP' )
            #Adjust plotting range of two y axes
            org1 = 0.0
            org2 = 0.0
            pos = 0.25
            align.yaxes(ax, org1, ax2, org2, pos)
            plt.margins(0)
            ax2.spines['right'].set_color('red')
            ax2.tick_params(axis='y', colors='red')
            ax2.tick_params(pad=2)
            if i!=3: ax2.set_yticklabels([])
            if j==1 and i==3: ax2.legend(loc=1,bbox_to_anchor=(0,0,1,0.85),frameon=False, prop={'size':8})
            count = count + 1
            
           
    # peak dF/F
    (peak_tuning, sem_tuning) = cal_peaktuning_sem(data_list, sem_list)
    for j in range(2):
        setmyaxes(4.47*xoffs+leftm, upperm-j*yoffs,xsize,ysize,(0,myylim[0][1]))
        #plot_data(peak_tuning[0][j*4:4+j*4],'k','Arclight')
        plot_data_witherrorbar(peak_tuning[0][j*4:4+j*4], sem_tuning[0][j*4:4+j*4], 'k', 'Arclight')
        if j==0: plt.title(r'Peak $\Delta$F/F')
        if j==1: plt.xlabel(myxlabel)
        if j==1: 
            ax=plt.gca()
            # get handles
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            plt.legend(handles, labels,loc=1,frameon=False, prop={'size':8})
        ax = plt.gca()
        ax2= ax.twinx()
        ax2.set_ylim((0,myylim[1][1]))
        plt.sca(ax2)
        #plot_data(peak_tuning[1][j*4:4+j*4],'r','GCaMP')
        plot_data_witherrorbar(peak_tuning[1][j*4:4+j*4], sem_tuning[1][j*4:4+j*4], 'r', 'GCaMP')
        
        ax2.spines['right'].set_color('red')
        ax2.tick_params(axis='y', colors='red')
        if j==1: 
            ax=plt.gca()
            # get handles
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            plt.legend(handles,labels,loc=1,bbox_to_anchor=(0,0,1,0.85),frameon=False, prop={'size':8})
        
        plt.xticks(range(0,4),xticklabel)
        
            
    #plt.text(220,1000*upperm+110,'Grating Speed',fontsize=12,transform=None)
    #plt.text(200,600*upperm+100,mytitle,fontsize=12,transform=None)
    
def plot_DSI_new(data,xlabel,xticklabel):
    
    xsize=0.20
    ysize=0.15
    xoffs=0.35
    yoffs=0.10
    leftm=0.20
    
    mytitle=['Grating', 'Edge']
    #data = [[T4_arclight_gratings_mean, T4_gcamp_gratings_mean], [T4_arclight_edge_mean, T4_gcamp_edge_mean]]
    for i in range(2):
        #dsi = cal_dsi(data[i])
        setmyaxes(i*xoffs+leftm,yoffs,xsize,ysize,myylim=(0.0,1.1))
        plot_data_witherrorbar(data[2*i][:,4], data[2*i][:,5], 'k', 'Arclight')
        plot_data_witherrorbar(data[2*i+1][:,4], data[2*i+1][:,5], 'r', 'GCaMP')
        #plot_data(data[2*i][:,4],'k','Arclight')
        #plot_data(data[2*i+1][:,4],'r','GCaMP')
        plt.xticks(range(0,4),xticklabel)
        plt.ylabel('DS-index')
        plt.xlabel(xlabel)
        plt.title(mytitle[i],fontsize=12)
        ax=plt.gca()
        # get handles
        handles, labels = ax.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        if i==1: plt.legend(handles,labels,loc=0,frameon=False, prop={'size':8})
        
        #plt.legend(loc=0,frameon=False,prop={'size':8})

def plot_DSI(data,xlabel,xticklabel):
    
    xsize=0.20
    ysize=0.15
    xoffs=0.35
    yoffs=0.10
    leftm=0.20
    
    mytitle=['Grating', 'Edge']
    #data = [[T4_arclight_gratings_mean, T4_gcamp_gratings_mean], [T4_arclight_edge_mean, T4_gcamp_edge_mean]]
    for i in range(2):
        dsi = cal_dsi(data[i])
        setmyaxes(i*xoffs+leftm,yoffs,xsize,ysize,myylim=(0.0,1.1))
        plot_data(dsi[0],'k','Arclight')
        plot_data(dsi[1],'r','GCaMP')
        plt.xticks(range(0,4),xticklabel)
        plt.ylabel('DS-index')
        plt.xlabel(xlabel)
        plt.title(mytitle[i],fontsize=12)
        plt.legend(loc=0,frameon=False,prop={'size':8})
        
def calc_peaktuning(data_list, sem_list, speed, contrast =False):
    peak_tuning = np.empty((len(data_list),len(speed),12)) #first dimesnion is for Arclight, Gcamp, Model respectively. 
                                #second dimension is for 4 speeds, 3rd dimension is for 12 directions
    peak_tuning_sem = np.empty((len(sem_list),len(speed),12))
    for j in range(len(data_list)):
        for i in range(4):
            rowindex = data_list[j][:,i:48+i:4].argmax(axis=0)
            colindex = np.arange(i,48+i,4)
            peak_tuning[j,i,:] = data_list[j][:,i:48+i:4].max(axis=0)
            peak_tuning_sem[j,i,:] = sem_list[j][rowindex,colindex]
    arc_max = peak_tuning[0].max()
    gcamp_max = peak_tuning[1].max()
    peak_tuning[0] = peak_tuning[0] / arc_max
    peak_tuning_sem[0] = peak_tuning_sem[0] / arc_max
    peak_tuning[1] = peak_tuning[1] / gcamp_max
    peak_tuning_sem[1] = peak_tuning_sem[1] / gcamp_max
    if contrast == True:
        peak_tuning = peak_tuning[:,::-1,:]
        peak_tuning_sem = peak_tuning_sem[:,::-1,:]
        
    return peak_tuning, peak_tuning_sem

def calc_Ldir(peak_tuning):
    Ldir_array = np.empty((peak_tuning.shape[0],peak_tuning.shape[1],2)) #first dimesnion is for Arclight, Gcamp, Model respectively. 
                                #second dimension is for 4 speeds, 3rd dimension is for angle and magnitude
    def Ldir(data):
        rot = np.arange(0.0,360.0,30.0)
        rot_rad = (rot*np.pi)/180
        data_x = np.sum(data*np.cos(rot_rad))
        data_y = np.sum(data*np.sin(rot_rad))
        angle_rad = np.arctan(data_y/data_x)
        if angle_rad <0.0:
            angle_rad = angle_rad+np.pi
        mag = np.sqrt(data_y**2 + data_x**2) / np.sum(data)
        return (angle_rad, mag)
    
    for j in range(peak_tuning.shape[0]):
        for i in range(peak_tuning.shape[1]):
            Ldir_array[j,i,:] = Ldir(peak_tuning[j,i,:])
    Ldir_array = np.nan_to_num(Ldir_array)
    
    return Ldir_array

def plot_newmodelsignals(data_list, sem_list, data_list1, myylim, upperm, mytitle, myxlabel, xticklabel,c,legend):
    
    xsize = 0.15
    ysize = 0.10
    xoffs = 0.167
    yoffs = 0.15
    leftm = 0.08
    
    #time-course of signals
    
    dt = 1/13.0
    x_values = np.arange(data_list[0].shape[0])*dt #changing x-axis to time. multiply with time step dt = 1/frequency
    count = 0 #count for stimuli
    
    for j in range(2):
        for i in range(4):
            setmyaxes(i*xoffs+leftm, upperm-j*yoffs,xsize,ysize,myylim)
            plt.xticks(np.arange(0.0,10.0,4.0))
            plot_timedata(x_values, data_list[0][:,count], sem_list[0][:,count],c[0],legend[0])
            plt.margins(0)
            ax = plt.gca()
            ax.set_ylim(myylim)
            ax.set_yticks(np.arange(myylim[0],myylim[1]-myylim[0],-myylim[0]))
            if i!=0: ax.set_yticklabels([])
            if j==0: ax.set_title(mytitle[i], fontsize=10)
            if j==1 and i==3: ax.legend(loc=1,frameon=False, prop={'size':8})
            if j==0 and i==0: ax.set_ylabel('PD', rotation=0,fontsize=10, labelpad=8)
            if j==1 and i==0: ax.set_ylabel('ND',rotation=0,fontsize=10, labelpad=8)
            
            plot_timedata(x_values, data_list[1][:,count], sem_list[1][:,count],c[1],legend[1])
            #if j==1 and i==3: ax.legend(loc=1,bbox_to_anchor=(0,0,1,0.85),frameon=False, prop={'size':8})
            if j==1 and i==3: ax.legend(loc=1,frameon=False, prop={'size':8})
            
            count = count + 1
            
    # peak dF/F
    #peak_tuning = cal_peaktuning(data_list)
    for j in range(2):
        setmyaxes(4.47*xoffs+leftm, upperm-j*yoffs,xsize,ysize,(0,myylim[1]))
        plot_data_witherrorbar(data_list1[0][:,2*j], data_list1[0][:,2*j+1], c[0], legend[0])
        
        #plot_data(peak_tuning[0][j*4:4+j*4],c[0],legend[0])
        if j==0: plt.title(r'Peak $\Delta$F/F')
        if j==1: plt.xlabel(myxlabel)
        if j==1: 
            ax=plt.gca()
            # get handles
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            plt.legend(handles, labels,loc=1,frameon=False, prop={'size':8})
        #if j==1: plt.legend(loc=1,frameon=False, prop={'size':8})
        ax = plt.gca()
        plot_data_witherrorbar(data_list1[1][:,2*j], data_list1[1][:,2*j+1], c[1], legend[1])
        #plot_data(peak_tuning[1][j*4:4+j*4],c[1],legend[1])
        if j==1: 
            ax=plt.gca()
            # get handles
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            plt.legend(handles, labels,loc=1,frameon=False, prop={'size':8})
        #if j==1: plt.legend(loc=1,frameon=False, prop={'size':8})
        
        plt.xticks(range(0,4),xticklabel)


def plot_modelsignals(data_list, sem_list, myylim, upperm, mytitle, myxlabel, xticklabel,c,legend):
    
    xsize = 0.15
    ysize = 0.10
    xoffs = 0.167
    yoffs = 0.15
    leftm = 0.08
    
    #time-course of signals
    
    dt = 1/13.0
    x_values = np.arange(data_list[0].shape[0])*dt #changing x-axis to time. multiply with time step dt = 1/frequency
    count = 0 #count for stimuli
    
    for j in range(2):
        for i in range(4):
            setmyaxes(i*xoffs+leftm, upperm-j*yoffs,xsize,ysize,myylim)
            plt.xticks(np.arange(0.0,10.0,4.0))
            plot_timedata(x_values, data_list[0][:,count], sem_list[0][:,count],c[0],legend[0])
            plt.margins(0)
            ax = plt.gca()
            ax.set_ylim(myylim)
            ax.set_yticks(np.arange(myylim[0],myylim[1]-myylim[0],-myylim[0]))
            if i!=0: ax.set_yticklabels([])
            if j==0: ax.set_title(mytitle[i], fontsize=10)
            if j==1 and i==3: ax.legend(loc=1,frameon=False, prop={'size':8})
            if j==0 and i==0: ax.set_ylabel('PD', rotation=0,fontsize=10, labelpad=8)
            if j==1 and i==0: ax.set_ylabel('ND',rotation=0,fontsize=10, labelpad=8)
            
            plot_timedata(x_values, data_list[1][:,count], sem_list[1][:,count],c[1],legend[1])
            #if j==1 and i==3: ax.legend(loc=1,bbox_to_anchor=(0,0,1,0.85),frameon=False, prop={'size':8})
            if j==1 and i==3: ax.legend(loc=1,frameon=False, prop={'size':8})
            
            count = count + 1
            
    # peak dF/F
    peak_tuning = cal_peaktuning(data_list)
    for j in range(2):
        setmyaxes(4.47*xoffs+leftm, upperm-j*yoffs,xsize,ysize,(0,myylim[1]))
        plot_data(peak_tuning[0][j*4:4+j*4],c[0],legend[0])
        if j==0: plt.title(r'Peak $\Delta$F/F')
        if j==1: plt.xlabel(myxlabel)
        if j==1: plt.legend(loc=1,frameon=False, prop={'size':8})
        ax = plt.gca()
        plot_data(peak_tuning[1][j*4:4+j*4],c[1],legend[1])
        if j==1: plt.legend(loc=1,frameon=False, prop={'size':8})
        
        plt.xticks(range(0,4),xticklabel)
        
def plot_Mi1Tm3signalsnew(data_list, sem_list , data_list1,myylim, upperm, mytitle, myxlabel, xticklabel,c,legend=False, tm3 = False):
    xsize = 0.15
    ysize = 0.10
    xoffs = 0.167
    yoffs = 0.15
    leftm = 0.08
    
    dt = 1/13.0
    x_values = np.arange(data_list[0].shape[0])*dt  #changing x-axis to time. multiply with time step dt = 1/frequency
    count = 0 #count for stimuli
    
    for i in range(4):
        setmyaxes(i*xoffs+leftm, upperm, xsize, ysize, myylim[0])
        plt.xticks(np.arange(0,10.0,4.0))
        ydata = data_list[0][:,count]
        semdata = sem_list[0][:,count]
        plot_timedata(x_values, ydata,semdata, c[0], 'Arclight')
        plt.margins(0)
        ax = plt.gca()
        if tm3 == False :
            ax.set_ylim(myylim[0])
            ax.set_yticks(np.arange(myylim[0][0],myylim[0][1]-myylim[0][0],-myylim[0][0]))
        else:
            ax.set_ylim(myylim[0])
            ax.set_yticks(np.arange(myylim[0][0], myylim[0][1]+0.05,0.05))
        if i!=0: ax.set_yticklabels([])
        ax.set_title(mytitle[i], fontsize=10)
        if legend == True and i==1 : ax.legend(loc=1, frameon=False, prop={'size':8})
            
        ax2 = ax.twinx()
        ax2.set_ylim(myylim[1])
        ax2.set_yticks(np.arange(myylim[1][0],myylim[1][1]-myylim[1][0],-myylim[1][0]))
        plt.sca(ax2)
        plot_timedata(x_values, data_list[1][:,count], sem_list[1][:,count],c[1], 'GCaMP' )
        
        org1=0.0
        org2=0.0
        if tm3 == False: 
            pos=0.25
        else:
            pos = 0.333
        align.yaxes(ax, org1, ax2, org2, pos)
        plt.margins(0)
        ax2.spines['right'].set_color('red')
        ax2.tick_params(axis='y', colors='red')
        ax2.tick_params(pad=2)
        if i!=3: ax2.set_yticklabels([])
        if legend == True and i==1: ax2.legend(loc=1,bbox_to_anchor=(0,0,1,0.85),frameon=False, prop={'size':8})
        count = count + 1
            
    # peak dF/F
    #peak_tuning = cal_peaktuning(data_list,ncol=4)
    # if tm3 == False : 
    #     setmyaxes(4.47*xoffs+leftm, upperm,xsize,ysize,(0,myylim[0][1]+0.06))
    # else:
    #     setmyaxes(4.47*xoffs+leftm, upperm,xsize,ysize,(0,myylim[0][1]+0.05))
    setmyaxes(4.47*xoffs+leftm, upperm,xsize,ysize,(0,myylim[2][1]))
    plot_data_witherrorbar(data_list1[0][:,0], data_list1[0][:,1], 'k', 'Arclight')
        
    #plot_data(peak_tuning[0][0:4],'k','Arclight')
    plt.title(r'Peak $\Delta$F/F')
    plt.xlabel(myxlabel)
    # if tm3==False and legend == True : 
    #     ax=plt.gca()
    #     handles, labels = ax.get_legend_handles_labels()
    #     handles = [h[0] for h in handles]
    #     plt.legend(handles, labels,loc=3,frameon=False, prop={'size':8}) 
        #plt.legend(loc=3,frameon=False, prop={'size':8})
    ax = plt.gca()
    ax2= ax.twinx()
    ax2.set_ylim((0,myylim[3][1]))
    # if tm3 == False : 
    #     ax2.set_ylim((0,myylim[1][1]+0.2))
    # else:
    #     ax2.set_ylim((0,myylim[1][1]))
    
    plt.sca(ax2)
    
    plot_data_witherrorbar(data_list1[1][:,0], data_list1[1][:,1], 'r', 'GCaMP')
    #plot_data(peak_tuning[1][0:4],'r','GCaMP')
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')
    # if tm3==False and legend == True : 
    #     ax=plt.gca()
    #     handles, labels = ax.get_legend_handles_labels()
    #     handles = [h[0] for h in handles]
    #     plt.legend(handles, labels,loc=3,bbox_to_anchor=(0,0.18,1.0,0.0),frameon=False, prop={'size':8})
    #     #plt.legend(loc=3,bbox_to_anchor=(0,0.18,1.0,0.0),frameon=False, prop={'size':8})
    
    
            
    
       
    plt.xticks(range(0,4),xticklabel)
    
        
        
def plot_Mi1Tm3signals(data_list, sem_list , myylim, upperm, mytitle, myxlabel, xticklabel,c,legend=False, tm3 = False):
    xsize = 0.15
    ysize = 0.10
    xoffs = 0.167
    yoffs = 0.15
    leftm = 0.08
    
    dt = 1/13.0
    x_values = np.arange(data_list[0].shape[0])*dt  #changing x-axis to time. multiply with time step dt = 1/frequency
    count = 0 #count for stimuli
    
    for i in range(4):
        setmyaxes(i*xoffs+leftm, upperm, xsize, ysize, myylim[0])
        plt.xticks(np.arange(0,10.0,4.0))
        ydata = data_list[0][:,count]
        semdata = sem_list[0][:,count]
        plot_timedata(x_values, ydata,semdata, c[0], 'Arclight')
        plt.margins(0)
        ax = plt.gca()
        if tm3 == False :
            ax.set_ylim(myylim[0])
            ax.set_yticks(np.arange(myylim[0][0],myylim[0][1]-myylim[0][0],-myylim[0][0]))
        else:
            ax.set_ylim(myylim[0])
            ax.set_yticks(np.arange(myylim[0][0], myylim[0][1]+0.05,0.05))
        if i!=0: ax.set_yticklabels([])
        ax.set_title(mytitle[i], fontsize=10)
        if legend == True and i==1 : ax.legend(loc=1, frameon=False, prop={'size':8})
            
        ax2 = ax.twinx()
        ax2.set_ylim(myylim[1])
        ax2.set_yticks(np.arange(myylim[1][0],myylim[1][1]-myylim[1][0],-myylim[1][0]))
        plt.sca(ax2)
        plot_timedata(x_values, data_list[1][:,count], sem_list[1][:,count],c[1], 'GCaMP' )
        
        org1=0.0
        org2=0.0
        if tm3 == False: 
            pos=0.25
        else:
            pos = 0.333
        align.yaxes(ax, org1, ax2, org2, pos)
        plt.margins(0)
        ax2.spines['right'].set_color('red')
        ax2.tick_params(axis='y', colors='red')
        ax2.tick_params(pad=2)
        if i!=3: ax2.set_yticklabels([])
        if legend == True and i==1: ax2.legend(loc=1,bbox_to_anchor=(0,0,1,0.85),frameon=False, prop={'size':8})
        count = count + 1
            
    # peak dF/F
    peak_tuning = cal_peaktuning(data_list,ncol=4)
    if tm3 == False : 
        setmyaxes(4.47*xoffs+leftm, upperm,xsize,ysize,(0,myylim[0][1]))
    else:
        setmyaxes(4.47*xoffs+leftm, upperm,xsize,ysize,(0,myylim[0][1]+0.05))
    plot_data(peak_tuning[0][0:4],'k','Arclight')
    plt.title(r'Peak $\Delta$F/F')
    plt.xlabel(myxlabel)
    if legend == True : plt.legend(loc=2,frameon=False, prop={'size':8})
    ax = plt.gca()
    ax2= ax.twinx()
    ax2.set_ylim((0,myylim[1][1]))
    plt.sca(ax2)
    plot_data(peak_tuning[1][0:4],'r','GCaMP')
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')
    if legend == True : plt.legend(loc=2,bbox_to_anchor=(0,0.82,1.0,0.0),frameon=False, prop={'size':8})
        
    plt.xticks(range(0,4),xticklabel)
 
def plot_Mi1Tm3modelsignalsnew(data_list, sem_list , data_list1, myylim, upperm, mytitle, myxlabel, xticklabel,c,legend=False, tm3 = False):
    xsize = 0.15
    ysize = 0.10
    xoffs = 0.167
    yoffs = 0.15
    leftm = 0.08
    
    dt = 1/13.0
    x_values = np.arange(data_list[0].shape[0])*dt  #changing x-axis to time. multiply with time step dt = 1/frequency
    count = 0 #count for stimuli
    
    for i in range(4):
        setmyaxes(i*xoffs+leftm, upperm, xsize, ysize, myylim[0])
        plt.xticks(np.arange(0,10.0,4.0))
        ydata = data_list[0][:,count]
        semdata = sem_list[0][:,count]
        plot_timedata(x_values, ydata,semdata, c[0], 'GCaMP')
        plt.margins(0)
        ax = plt.gca()
        ax.set_ylim(myylim[0])
        ax.set_yticks(np.arange(myylim[0][0],myylim[0][1]-myylim[0][0],-myylim[0][0]))
        #ax.set_ylim(myylim[0])
        #ax.set_yticks(np.arange(myylim[0],myylim[1]-myylim[0],-myylim[0]))
        
        plot_timedata(x_values,data_list[1][:,count], sem_list[1][:,count], c[1], 'Model')
        
        
        
        if i!=0: ax.set_yticklabels([])
        ax.set_title(mytitle[i], fontsize=10)
        if legend == True and i==3 : ax.legend(loc=1, frameon=False, prop={'size':8})
        count+=1
            
        
            
    # peak dF/F
    #peak_tuning = cal_peaktuning(data_list,ncol=4)
    setmyaxes(4.47*xoffs+leftm, upperm,xsize,ysize,(0,myylim[1][1]))
    plot_data_witherrorbar(data_list1[0][:,0], data_list1[0][:,1], c[0], 'GCaMP')
    #plot_data(peak_tuning[0][0:4],c[0],'GCaMP')
    plot_data_witherrorbar(data_list1[1][:,0], data_list1[1][:,1], c[1], 'Model')
    #plot_data(peak_tuning[1][0:4],c[1],'Model')
    plt.title(r'Peak $\Delta$F/F')
    plt.xlabel(myxlabel)
    #if legend==True: plt.legend(loc=0,frameon=False, prop={'size':8})    
    plt.xticks(range(0,4),xticklabel)
    
def plot_Mi1Tm3modelsignals(data_list, sem_list , myylim, upperm, mytitle, myxlabel, xticklabel,c,legend=False, tm3 = False):
    xsize = 0.15
    ysize = 0.10
    xoffs = 0.167
    yoffs = 0.15
    leftm = 0.08
    
    dt = 1/13.0
    x_values = np.arange(data_list[0].shape[0])*dt  #changing x-axis to time. multiply with time step dt = 1/frequency
    count = 0 #count for stimuli
    
    for i in range(4):
        setmyaxes(i*xoffs+leftm, upperm, xsize, ysize, myylim)
        plt.xticks(np.arange(0,10.0,4.0))
        ydata = data_list[0][:,count]
        semdata = sem_list[0][:,count]
        plot_timedata(x_values, ydata,semdata, c[0], 'GCaMP')
        plt.margins(0)
        ax = plt.gca()
        ax.set_ylim(myylim[0])
        ax.set_yticks(np.arange(myylim[0],myylim[1]-myylim[0],-myylim[0]))
        
        plot_timedata(x_values,data_list[1][:,count], sem_list[1][:,count], c[1], 'Model')
        
        
        
        if i!=0: ax.set_yticklabels([])
        ax.set_title(mytitle[i], fontsize=10)
        if legend == True and i==3 : ax.legend(loc=1, frameon=False, prop={'size':8})
        count+=1
            
        
            
    # peak dF/F
    peak_tuning = cal_peaktuning(data_list,ncol=4)
    setmyaxes(4.47*xoffs+leftm, upperm,xsize,ysize,(0,myylim[1]))
    plot_data(peak_tuning[0][0:4],c[0],'GCaMP')
    plot_data(peak_tuning[1][0:4],c[1],'Model')
    plt.title(r'Peak $\Delta$F/F')
    plt.xlabel(myxlabel)
    if legend==True: plt.legend(loc=0,frameon=False, prop={'size':8})    
    plt.xticks(range(0,4),xticklabel)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        