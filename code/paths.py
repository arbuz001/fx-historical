'''
Contains paths to data and submission folders
'''
import os
import sys

from pprint import pprint

# all project foldres consolidated

# 0. print all current sys.paths (i.e. locations where python looks for modules)
pprint(sys.path)

strPrjPath = 'c:/works-and-documents/svn/kaggle/data-fx-historical'

strDataPath = strPrjPath + '/data/'
strDataCode = strPrjPath + '/code/'

sys.path.insert(0, strDataPath)
sys.path.insert(0, strDataCode)

# 1. two new sys.paths to look for modules should be added 
pprint(sys.path)
