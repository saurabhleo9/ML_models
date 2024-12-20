
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot
import os
import pandas as pa
import numpy as np

 
# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]
 
# fit an random forest model and make a one step prediction
def xgboost_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		#print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions
 
# load the dataset
#series = read_csv("/media/leo9/Leo9/z_engineered_data/seriescsvcimaxT_all40_d.nc.csv.csv", header=0, index_col=0)

filez = os.listdir("C:/Users/D.N. singh/Desktop/a_paper_rf_chek/")
pathz = ("C:/Users/D.N. singh/Desktop/a_paper_rf_chek/")

valz=[]

for a in range(len(filez)) :
    series = read_csv(pathz+filez[a], header=0, index_col=0, parse_dates=True)
    #series = read_csv("/media/leo9/Leo9/z_engineered_data/series_csvcimaxT_all40_d.nc.csv.csv", header=0, index_col=0, parse_dates=True)
    series1=series.dropna()
    series2=series1.copy()
    values = series1.values
    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=6)
    # evaluate
    mae, y, yhat = walk_forward_validation(data, 840)
    obser=series2.tail(840)
    pre=yhat.copy()
    obser['xgb']=pre
    filo = os.path.join('C:/Users/D.N. singh/Desktop/xgb_60y/', 'xgb_'+filez[a]+'.csv')
    valz.append(obser)
    valz[a].to_csv(filo, index=True)
    print("Done")
   