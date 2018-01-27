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

def fnWriteIntermediateLog(Z, strLogFile = ""):
#	
# fnWriteIntermediateLog stores intermediate output to CSV file
#
	if strLogFile != "":
		strDir = os.path.dirname(strLogFile)
		try:
			os.stat(strDir)
		except:
			os.mkdir(strDir) 
		
		qDF = pd.DataFrame(Z)
		qDF.to_csv(strLogFile, index = False)
