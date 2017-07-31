import datetime
import os
import itertools as it
import shutil
#import json_tricks
import copy
import numpy as np
import pandas as pd
import math 
import IPython as ipd
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
idx = pd.IndexSlice


digits = 3
pd.options.display.chop_threshold = 10**-(digits+1)
#pd.options.display.precision = digits
pd.options.display.float_format = lambda x: '{0:.{1}f}'.format(x,digits)
pd.options.display.show_dimensions = True
def display(X):
    if isinstance(X, pd.Series) or (isinstance(X, np.ndarray) and X.ndim <=2):
        ipd.display.display(pd.DataFrame(X))
    else:
        ipd.display.display(X)

import matplotlib.pyplot as plt
plt.style.use("classic")
plt.style.use("fivethirtyeight")
#plt.style.use("bmh")
plt.rc("figure", figsize=(5,3))

def write_json(filename, obj):
    with open(filename,'w') as f:
        json_tricks.dump(obj, f, indent=3)

def read_json(filename):
    with open(filename,'r') as f:
        return json_tricks.load(f)

def Lnorm(v,p=2):
    if p%2 == 0:
        return (v**p).sum()**(1/p)
    else:
        return (v.abs()**p).sum()**(1/p)
    
def Ldist(v,w,p=2):
    return Lnorm(v-w,p)
