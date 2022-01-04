#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:28:33 2021

@author: leo9
"""

from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset as nc
import pandas as pa
import pymannkendall as mkt
from netCDF4 import num2date
import matplotlib.pyplot as plot
import numpy as np
import scipy.stats.mstats as st
import numpy.ma as ma
from matplotlib.patches import Path, PathPatch
from matplotlib import colors
from mpl_toolkits.axes_grid1 import AxesGrid
import os

val=[]

fii=os.listdir("/home/leo9/Desktop/tren/slop_plt/")
path="/home/leo9/Desktop/tren/slop_plt/"

for a in range(len(fii)) : 
    file = nc(path+fii[a])
    
    c1=file.variables['lat'][:]
    b1=file.variables['lon'][:]
    timme=file.variables['time'][:]
    ro=file.variables['temp'][:]



    times = num2date(timme, (file.variables['time']).units)
    times_grid, latitudes_grid, longitudes_grid = [x.flatten() for x in np.meshgrid(times, c1, b1, indexing='ij')]
    df = pa.DataFrame({
        'time': times_grid,
        'latitude': latitudes_grid,
        'longitude': longitudes_grid,
        'tem': ro.flatten()})

    del df["time"]
    chh_df = df[(df['tem'] > 0.04)]
    chh_neg = df[(df['tem'] < -0.04)]

    color = [(0.0,0.0,0.6,0.99),(0.0,0.0,0.6,0.99),(0.0,0.0,0.8,0.9),(0.0,0.0,0.67,0.55),(0.67,0.0,0.0,0.55),(0.8,0.0,0.0,0.9),(0.6,0.0,0.0,0.99),(0.6,0.0,0.0,0.99)]
    levv=[-0.12,-0.08,-0.04,-0.032,0,0.032,0.04,0.08,0.12]


    fig = plot.figure(num=None, figsize=(12, 8) )
    ax = Basemap(projection='mill',llcrnrlat=c1.min(),urcrnrlat=c1.max(),llcrnrlon=b1.min(),urcrnrlon=b1.max(),resolution='c')      
    x,y=ax(*np.meshgrid(b1,c1))
    ax.contourf(x,y,ro[0,:,:], shading='flat',levels=levv, colors=color)
    ax.colorbar(location='right')


    tak_lat=chh_df["latitude"]
    tak_lon=chh_df["longitude"]

    fin_lat=tak_lat.to_numpy(dtype='float32')
    fin_lon=tak_lon.to_numpy(dtype='float32')

    neg_lat=chh_neg["latitude"]
    neg_lon=chh_neg["longitude"]

    fin_neg_lat=neg_lat.to_numpy(dtype='float32')
    fin_neg_lon=neg_lon.to_numpy(dtype='float32')


    x_mes, y_mes = ax(fin_lon, fin_lat)
    x_ne, y_ne = ax(fin_neg_lon, fin_neg_lat)

    ax.scatter(x_mes,y_mes, 40, marker='.', color='k')
    ax.scatter(x_ne, y_ne, 40, marker='.', color='k')

    ax.drawcoastlines(linewidth=2)
    ax.drawcountries(linewidth=2)
    ax.drawmapboundary(linewidth=2)



    ax.drawparallels(np.arange(10,40,5),labels=[1,0,0,0], fontsize=10, linewidth=0.0)
    ax.drawmeridians(np.arange(65,100,5),labels=[0,0,0,1], rotation=45, fontsize=10, linewidth=0.0)



    #plot.show()
    
    val.append(plot)
    val[a].savefig(path+"trrr"+fii[a]+".png", format="png", dpi=500)
    val[a].close()