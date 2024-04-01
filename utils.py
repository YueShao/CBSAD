from cdt.independence.stats import AdjMI


import os
import sys
import time
import matplotlib.pyplot as plt
import math

sys.path.append("")
import unittest

import numpy as np
import pandas as pd

# -*- coding: utf-8 -*-


import os
import sys
from time import time
import scipy.stats as ss

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.utils.utility import standardizer
from pyod.models.pca import PCA

from pyod.models.knn import KNN
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score




def get_cormat_data(X_train_norm):
    obj = AdjMI()
    i,j= X_train_norm.shape
#     print(i,j)
    cormat=np.zeros((j,j))
#     print(cormat)
    for m in range(j):
        for n in range(m+1,j):
#             print(m,n)
            if np.var(X_train_norm[:,m])==0 or np.var(X_train_norm[:,n])==0:
                score = 0
            else:
                score=obj.predict(X_train_norm[:,m],X_train_norm[:,n])
            cormat[m,n]=score
#     print(np.mean(cormat))
    return cormat



    
def DFS(index, lst,matrix, keyList):

    if index not in keyList:
        keyList.append(index)
        lst.append(index)
        _,length = matrix.shape
        for i in range(length):
            if matrix[index][i] != 0 and i not in keyList:
                DFS(i, lst, matrix,keyList)


def find_groups(matrix,k):
    oldconnect=cormatfilter(matrix,k)
    
    keyList=[]
    _,length = matrix.shape
    groups=[]
    for i in range(length):
        lst=[]
        DFS(i,lst,oldconnect,keyList)
        if len(lst)!=0:
            groups.append(lst)
    return oldconnect,groups


            
def cormatfilter(matrix,k):
    matrix=abs(matrix)
    print(matrix.max())
    thresh = matrix.max() / float(k)
#     thresh = 0.4
    i,j = matrix.shape
    oldconnect=np.zeros((j,j))
    for m in range(i):
        for n in range(m+1,j):
            if(matrix[m,n]>thresh):
                oldconnect[m,n]=1
                oldconnect[n,m]=1
    return oldconnect



def sortgroup(groups):
#     groups.sort(key = lambda i:len(i),reverse=True)
    groups.sort(key = lambda i:len(i),reverse=False)
    return groups

