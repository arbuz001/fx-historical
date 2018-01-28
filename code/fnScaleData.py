import numpy as np

def fnScaleData(gZ):

	gZScaled = (gZ - np.mean(gZ))/np.std(gZ) 

	return gZScaled
