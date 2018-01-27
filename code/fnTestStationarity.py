from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from matplotlib.pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd



def fnTestStationarity(timeseries):
    
	#Determing rolling statistics
	rolmean	= timeseries.rolling(window = 12,center=False).mean()
	rolstd	= timeseries.rolling(window = 12,center=False).std()

	#Plot rolling statistics:
	orig	= plt.plot(timeseries, color = 'blue',label = 'original')
	mean	= plt.plot(rolmean, color = 'red', label = 'rolling mean')
	std		= plt.plot(rolstd, color = 'black', label = 'rolling std')
	
	plt.legend(loc = 'best')
	plt.title('rolling mean & standard deviation')
	plt.show(block = False)

	#Perform Dickey-Fuller test:
	print 'results of dickey-fuller test:'
	dftest		= adfuller(timeseries, autolag='AIC')
	dfoutput	= pd.Series(dftest[0:4], index=['test statistic','p-value','#lags used','number of observations used'])
	
	for key,value in dftest[4].items():
		dfoutput['critical value (%s)'%key] = value
	
	print dfoutput
