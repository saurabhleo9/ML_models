#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:58:23 2021

@author: leo9
"""

import pandas as pa
import iris
import iris.plot as iplt
import cartopy.crs as ccrs
import cartopy.feature as fea
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from eofs.iris import Eof
from netCDF4 import Dataset as nc
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

val=[]

fii=os.listdir("/media/leo9/Leo9/IMD/IMD_rf/all_new/")
path="/media/leo9/Leo9/IMD/IMD_rf/all_new/"
path1="/media/leo9/Leo9/IMD/IMD_rf/eof2/"
fname = '/home/leo9/Desktop/shapes/india_st.shp'

#fii=os.listdir("/home/leo9/Desktop/for_eof/")
#path="/home/leo9/Desktop/for_eof/"

for a in range(len(fii)) : 
    data = iris.load_cube(path+fii[a])
    
    data_mean = data.collapsed('time', iris.analysis.MEAN)
    data_ano = data - data_mean

    # Create an EOF solver to do the EOF analysis. Square-root of cosine of
    # latitude weights are applied before the computation of EOFs.
    solver = Eof(data_ano, weights='coslat') 
    #solver = Eof(data, weights='coslat')
    # Retrieve the leading EOF, expressed as the covariance between the leading PC
    # time series and the input SLP anomalies at each grid point.
    eof1 = solver.eofsAsCovariance(neofs=3) 

    pcs = solver.pcs(npcs=3, pcscaling=1)
    pcs_test = solver.pcs(npcs=3, pcscaling=0)
    frac_var = solver.varianceFraction(neigs=3)
    total_var = solver.totalAnomalyVariance()
    eigenvalue = solver.eigenvalues(neigs=3)

    errors = solver.northTest(neigs=3, vfscaled=True)
    reconstruction = solver.reconstructedField(3)
    see=frac_var[1].data*100
    yy="%"
    oo=("%.2f %s"%(see, yy))
    #result = data.collapsed('time', iris.analysis.STD_DEV)
    #data_mean = data.collapsed('time', iris.analysis.MEAN)
    #cvv=result/data_mean
    proj = ccrs.Mercator(central_longitude=0.0, min_latitude=7.5, max_latitude=37.5, globe=None, latitude_true_scale=None, false_easting=0.0, false_northing=0.0, scale_factor=None)
    ax = plt.axes(projection=proj)
    
    shape_feature = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), 
                               linewidth = 1.4, facecolor = (1, 1, 1, 0), 
                               edgecolor = (0, 0, 0, 1))

    ax.add_feature(shape_feature)
    
    pic=iplt.contourf(eof1[1,:,:], cmap=plt.cm.jet)
    pic
    ax.coastlines()
    ax.set_xticks([65, 75, 85, 95], crs=ccrs.PlateCarree())
    ax.set_yticks([10, 15, 20, 25, 30, 35], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.1f',
                                       degree_symbol='',
                                       dateline_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.1f',
                                      degree_symbol='')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.add_feature(fea.BORDERS)
    ax.add_feature(fea.OCEAN, zorder=100, edgecolor='k', facecolor='w')
    plt.colorbar(pic, ax=ax, orientation="vertical", pad=.05)
    plt.title(oo)
    val.append(plt)
    val[a].savefig(path1+"eof2"+fii[a]+".png", format="png", dpi=1000)
    val[a].close()