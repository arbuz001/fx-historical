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
