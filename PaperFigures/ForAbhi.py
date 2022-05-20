# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:05:59 2021

@author: aborst
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
def setmyaxes_polar(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize],polar=True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
def plot_data():
    
    myt=np.arange(800)*0.01
    myy=np.sin(myt*2*np.pi)
    myy[0:200]=0
    myy[600:800]=0
    
    plt.plot(myt,myy)
    
def plot_signals(upperm,mytitle):
    
    xsize=0.15
    ysize=0.10
    xoffs=0.16
    yoffs=0.15
    leftm=0.07
    
    # time-course of signals
    
    for i in range(4):
        for j in range(2):
            setmyaxes(i*xoffs+leftm,upperm-j*yoffs,xsize,ysize)
            plot_data()
    
    # peak DF/F
    
    for j in range(2):
        setmyaxes(4.5*xoffs+leftm,upperm-j*yoffs,xsize,ysize)
        if j==0: plt.title('Peak DF/F')
    
    plt.text(220,1000*upperm+110,mytitle,fontsize=12,transform=None)
    
def plot_polars(upperm,mytitle):
    
    xsize=0.20
    ysize=0.15
    xoffs=0.22
    leftm=0.07
    
    for i in range(4):
        setmyaxes_polar(i*xoffs+leftm,upperm-0.1,xsize,ysize)
    
    plt.text(300,1000*upperm+110,mytitle,fontsize=12,transform=None)
    
def plot_DSI(xlabel):
    
    xsize=0.20
    ysize=0.15
    xoffs=0.35
    yoffs=0.10
    leftm=0.20
    
    mytitle=['Grating','Edge']
    
    for i in range(2):
        setmyaxes(i*xoffs+leftm,yoffs,xsize,ysize)
        plt.ylabel('DS-index')
        plt.xlabel(xlabel)
        plt.title(mytitle[i],fontsize=12)
        
def plot_Ldir(xlabel):
    
    xsize=0.20
    ysize=0.15
    xoffs=0.35
    yoffs=0.10
    leftm=0.20
    
    mytitle='Grating'
    xlabel=['speed [deg/s]','contrast[%]']
    
    for i in range(2):
        setmyaxes(i*xoffs+leftm,yoffs,xsize,ysize)
        plt.ylabel('Ldir-index')
        plt.xlabel(xlabel[i])
        plt.title(mytitle,fontsize=12)
                          
def plot_fig1():
    
    plt.figure(figsize=(7.5,10))
    
    plot_signals(0.85,'Grating Speed ->')
    plot_signals(0.50,'  Edge Speed ->')
    plot_DSI('speed [deg/s]')
    
def plot_fig2():
    
    plt.figure(figsize=(7.5,10))
    
    plot_signals(0.85,'Grating Contrast ->')
    plot_signals(0.50,'  Edge Contrast ->')
    plot_DSI('contrast [%]')
   
def plot_fig3():
    
    plt.figure(figsize=(7.5,10))
    
    plot_polars(0.85,'Grating Speed ->')
    plot_polars(0.50,'Grating Contrast ->')
    plot_Ldir('contrast [%]')
    
#plot_fig1()
#plot_fig2()
plot_fig3()