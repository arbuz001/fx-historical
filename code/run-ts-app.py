'''
Contains code to explore different methods for financial time series
https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
'''
import matplotlib.pylab as plt
import mlpy
import numpy as np
import os
import pandas as pd

from datetime import datetime
from matplotlib.pylab import rcParams
from scipy.stats.stats import pearsonr
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller

# import custom functions from custom module
from fnTestStationarity import *
from fnScaleData import *
from fnUnScaleData import *
from fnWriteIntermediateLog import *

bPlot	= True
bLog	= True
# all project foldres are already defined in init-all.py

# loading and handling data
strDateFormat	= '%Y-%m-%d'
dateparse		= lambda dates: pd.datetime.strptime(dates, strDateFormat)

strFileIn	= strDataPath + 'USD.RUB.trimmed.csv'
data 		= pd.read_csv(strFileIn, parse_dates = ['YYYY-MM-DD'], index_col = ['YYYY-MM-DD'], date_parser = dateparse)
# print '\n Data Types:'
# print data.dtypes
print data.head()

data.index
gSP350Eur = data['RUB-USD']
gSP350Eur.head(10)

# example on how to use distance between time series
# x = [0,0,0,0,1,1,2,2,3,2,1,1,0,0,0,0]
# y = [0,0,1,1,2,2,3,3,3,3,2,2,1,1,0,0]
# dist, cost, path = mlpy.dtw_std(x, y, dist_only=False)
# dist, cost, path = mlpy.dtw_std(gSP350Eur, gREA, dist_only=False)

# access particular data points
'''
	#1. Specific the index as a string constant:
	gSP350Eur['2015-10-01']
	#2. Import the datetime library and use 'datetime' function:
	gSP350Eur[datetime(2015,5,7)]
	#3. Specify the entire range:
	gSP350Eur['2015-05-06':'1949-05-01']
	#4. Use ':' if one of the indices is at ends:
	gSP350Eur[:'1949-05-01']
	#5. Whole year 
	gSP350Eur['2015']
'''

# mean-std scaled original time series
gSP350EurScaled	= fnScaleData(gSP350Eur)

# # to go back to original unscaled time series
# rMean	= np.mean(gSP350Eur)
# rStd	= np.std(gSP350Eur)
# gSP350EurOriginal = fnUnScaleData(fnScaleData(gSP350Eur),rMean,rStd)

if bLog:	
	strLogFile = strOutPath + " gSP350EurScaled.csv"
	fnWriteIntermediateLog(gSP350EurScaled.values, strLogFile)	

# # output correlation coefficient
# rCorr = np.corrcoef(gSP350EurSeasonal.values, gREASeasonal.values)
# rCorr = pearsonr(gSP350EurTrend.values, gREATrend.values)[0]

# check if time series is stationary
nStep = 12
for iWindow in range(1, nStep*6, nStep):
	gSP350EurMovingAverage = gSP350EurScaled.rolling(window = iWindow,center=False).mean()
	plt.plot(gSP350EurMovingAverage)

if bPlot:
	plt.title("rolling mean values with the step '" + str(nStep) + "' to limit of '" + str(nStep*6) + "'")
	plt.show()

# apply the transformation function to make time series stationary 
# gSP350EurTransformed = np.log(gSP350EurScaled)
# gSP350EurTransformed = np.sqrt(gSP350EurScaled)
gSP350EurTransformed = gSP350EurScaled

if bPlot: 
	plt.plot(gSP350EurScaled)
	plt.plot(gSP350EurTransformed, color='red')
	plt.show()

# smoothing by applying moving average	
'''	
gSP350EurTransformedMA = gSP350EurTransformed.rolling(window = 12).mean()
if bPlot: 
	plt.plot(gSP350EurTransformed)
	plt.plot(gSP350EurTransformedMA, color='red')
'''

# simple moving average, fixed number of lags
'''
gSP350EurTransformedMADiff = gSP350EurTransformed - gSP350EurTransformedMA
gSP350EurTransformedMADiff.dropna(inplace=True)

if bPlot: 
	plt.plot(gSP350EurTransformedMADiff)

gSP350EurTransformedMADiff.head(12)
fnTestStationarity(gSP350EurTransformedMADiff)
'''

# exponentially weighted moving average
'''
nStep = 12
for iHalfLife in range(1, nStep*6, nStep):
	gSP350EurTransformedExpWMA = gSP350EurTransformed.ewm(halflife = iHalfLife,ignore_na = False, min_periods = 0, adjust=True).mean()
	plt.plot(gSP350EurTransformedExpWMA)
	
if bPlot:
	plt.title("exp weighted MA with the variable half life in step '" + str(nStep) + "' to limit of '" + str(nStep*6) + "'")
	plt.show()
'''

'''
gSP350EurTransformedExpWMA = pd.ewma(gSP350EurTransformed, halflife = 12)
if bPlot:
	plt.plot(gSP350EurTransformed)
	plt.plot(gSP350EurTransformedExpWMA, color='red')

gSP350EurTransformedExpWMADiff = gSP350EurTransformed - gSP350EurTransformedExpWMA
gSP350EurTransformedExpWMADiff.dropna(inplace=True)

if bPlot: 
	plt.plot(gSP350EurTransformedExpWMADiff)

	fnTestStationarity(gSP350EurTransformedExpWMADiff)
'''

# decomposing time series
gSP350EurTransformedDecompose = seasonal_decompose(gSP350EurTransformed, freq = 12)

gSP350EurTrend		= gSP350EurTransformedDecompose.trend
gSP350EurSeasonal	= gSP350EurTransformedDecompose.seasonal
gSP350EurResidual	= gSP350EurTransformedDecompose.resid

if bPlot:
	plt.subplot(411)
	plt.plot(gSP350EurTransformed, label='Original')
	plt.legend(loc='best')
	plt.subplot(412)
	plt.plot(gSP350EurTrend, label='Trend')
	plt.legend(loc='best')
	plt.subplot(413)
	plt.plot(gSP350EurSeasonal,label='Seasonality')
	plt.legend(loc='best')
	plt.subplot(414)
	plt.plot(gSP350EurResidual, label='Residuals')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.show()	

# apply differencing
gSP350EurTransformedDiff = gSP350EurTransformed - gSP350EurTransformed.shift()
gSP350EurTransformedDiff.dropna(inplace=True)

gT = gSP350EurTransformedDiff.shift()
gT.dropna(inplace=True)

pearsonr((gSP350EurTransformedDiff.values)[1:], gT.values)[0]


if bPlot:
	plt.plot(gSP350EurTransformedDiff)
	plt.show()

fnTestStationarity(gSP350EurTransformedDiff)

# forecasting a time series using ARIMA
gLagACF		= acf(gSP350EurTransformed, nlags = 20)
# gLagACF		= acf(gSP350EurTransformedDiff, nlags = 20)

gLagPACF	= pacf(gSP350EurTransformed, nlags = 20, method = 'ols')
# gLagPACF	= pacf(gSP350EurTransformedDiff, nlags = 20, method = 'ols')

# # output correlation coefficient
# rCorr = np.corrcoef(gSP350EurTransformed.values, gSP350EurTransformed.values)
# rCorr = pearsonr(gSP350EurTransformed.values, gSP350EurTransformed.values)[0]


#Plot ACF: 
plt.subplot(121) 
plt.plot(gLagACF)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(gSP350EurTransformedDiff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(gSP350EurTransformedDiff)),linestyle='--',color='gray')
plt.title('autocorrelation function')

#Plot PACF:
plt.subplot(122)
plt.plot(gLagPACF)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(gSP350EurTransformedDiff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(gSP350EurTransformedDiff)),linestyle='--',color='gray')
plt.title('partial autocorrelation function')

plt.tight_layout()
plt.show()

# # option-1
# model = ARIMA(gSP350EurTransformed, order=(2, 1, 0))  
# results_AR = model.fit(disp=-1)  
# plt.plot(gSP350EurTransformedDiff)
# plt.plot(results_AR.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-gSP350EurTransformedDiff)**2))

# option-2
# model = ARIMA(gSP350EurTransformed, order=(2, 1, 2))
model = ARIMA(gSP350EurTransformed, order=(2, 1, 0))
results_ARIMA = model.fit(disp=-1)  
plt.plot(gSP350EurTransformedDiff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.6f'% sum((results_ARIMA.fittedvalues-gSP350EurTransformedDiff)**2))

# taking it back to original scale
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print predictions_ARIMA_diff.head(10)

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print predictions_ARIMA_diff_cumsum.head()
# plt.plot(predictions_ARIMA_diff_cumsum)

predictions_ARIMA_log = pd.Series(gSP350EurTransformed.ix[0], index=gSP350EurTransformed.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

# predictions_ARIMA = np.exp(predictions_ARIMA_log)
predictions_ARIMA = np.square(predictions_ARIMA_log)
plt.plot(gSP350Eur)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-gSP350Eur)**2)/len(gSP350Eur)))
