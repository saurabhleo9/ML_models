#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 02:44:36 2022

@author: leo9
"""

# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import iris.pandas as ip
import pandas as pa
import numpy as np


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset (column 3 for temperature values)
#dataframe = read_csv('/media/leo9/Leo9/z_engineered_data/series_csvcimaxT_all40_d.nc.csv.csv', usecols=[1], engine='python')


filez = os.listdir("/media/leo9/Leo9/IMD/IMD_rf/zzz/")
pathz = ("/media/leo9/Leo9/IMD/IMD_rf/zzz/")

valz=[]

for a in range(len(filez)) :
    dataframe = read_csv(pathz+filez[a], header=0, index_col=0, parse_dates=True)
    #series = read_csv("/media/leo9/Leo9/z_engineered_data/series_csvcimaxT_all40_d.nc.csv.csv", header=0, index_col=0, parse_dates=True)
    dataframe1 = dataframe.dropna()
    dataframe3 = dataframe1.copy()
    dataframe3 = dataframe3[:-1]
    #final_obs=dataframe3.tail(478)

    dataset = dataframe3.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.75)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    reshaped_testX=testX.copy()
    reshaped_testY=testY.copy()
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    pred_test=testPredict.copy()
    pred_train=trainPredict.copy()
    obs_train=trainY.copy()
    obs_test=testY.copy()
    
    final_obs=dataframe3.tail(len(pred_test))
    final_obs['LSTM']=pred_test

    filo = os.path.join('/media/leo9/Leo9/IMD/IMD_rf/z_lstm_output/', 'LSTM_'+filez[a]+'.csv')
    valz.append(final_obs)
    valz[a].to_csv(filo, index=True)
       
    print("Done")







#dataframe = read_csv('/media/leo9/Leo9/z_engineered_data/series_csvcimaxT_all40_d.nc.csv.csv', header=0, index_col=0, parse_dates=True)
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
#trainPredictPlot = numpy.empty_like(dataset)
#trainPredictPlot[:, :] = numpy.nan
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
#testPredictPlot = numpy.empty_like(dataset)
#testPredictPlot[:, :] = numpy.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()