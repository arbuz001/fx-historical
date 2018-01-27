'''
Contains paths to data and submission folders
'''

strPrjPath = '/home/alex/Documents/kaggle/data-fx-historical'

import os
import sys
import rlcompleter
import readline

from pprint import pprint

# 0. add autocomplete by tab feature
readline.parse_and_bind("tab: complete")

# 1. print all current sys.paths (i.e. locations where python looks for modules)
pprint(sys.path)

# all project foldres consolidated
strDataPath = strPrjPath + '/data'
strDataCode = strPrjPath + '/code'

sys.path.insert(0, strDataPath)
sys.path.insert(0, strDataCode)

# 2. two new sys.paths to look for modules should be added 
pprint(sys.path)
