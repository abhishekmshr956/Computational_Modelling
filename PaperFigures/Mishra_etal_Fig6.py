# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:05:59 2021

@author: aborst
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# T4 multiplicative model 

T4_param_label=['trld','TC hp [s]','TC lp1 [s]','TC lp1 [s]','gain [*100]']
T4_param=np.zeros((2,10))
T4_param[0]=[-0.34062, 0.00229,	0.90825, 0.00223, 2.58857, 0.03446, 2.67023, 0.02478, 143.1173,	0.99523]
T4_param[1]=[-0.24189, 0.00848,	0.35808, 0.01031, 1.40611, 0.09302, 2.0017,  0.15156, 132.92563,2.37975]

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
def plot_subfig(data,label,cell_label,cell_color):
    
    mywidth=0.36
    
    mydim=5
    
    param_data=data[0:10:2]
    param_data[4]=param_data[4]/100.0
    param_error=data[1:10:2]
    param_error[4]=param_error[4]/100.0

    plt.bar(np.arange(mydim),param_data,yerr=param_error,width=mywidth,color=cell_color,label=cell_label)
    plt.xticks(np.arange(mydim),label)
    plt.plot([-0.5,mydim-0.5],[0,0],linestyle='dashed',color='black')
    plt.ylabel('parameter value')
    plt.legend(loc=2,frameon=False,fontsize=14)
    plt.ylim(-0.5,3)
    
def plot_fig6():
    
    plt.figure(figsize=(12,6))
    
    cell_label=['GCaMP6f','GCaMP8f']
    cell_color=['green','orange']
    
    for i in range(2):
            
        plt.subplot(1,2,i+1)
        
        plot_subfig(T4_param[i],T4_param_label,cell_label[i],cell_color[i])
        
plot_fig6()
    