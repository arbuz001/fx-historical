import numpy as np

def fnUnScaleData(gZScaled,rMean,rStd):

	gZ = gZScaled*rStd + rMean 

	return gZ
