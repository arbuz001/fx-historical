def fnScaleData(gZ):

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

	gZScaled = (gZ - np.mean(gZ))/np.std(gZ) 
	
	print gZScaled
	
	return gZScaled
