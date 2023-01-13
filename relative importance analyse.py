#coding:utf-8
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import ImportData as Ds
import random
import numpy as np
from sklearn import linear_model
import math
import pandas as pd

###### the scikit-learning regression model
from sklearn.neural_network import MLPRegressor     # neural_network
from sklearn.gaussian_process import GaussianProcessRegressor   # gaussian process regression
from sklearn.neighbors import KNeighborsRegressor   # neighbors regression
from sklearn import svm # SVR
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn.linear_model import LinearRegression #linear regrssion(least square regression)
from sklearn.linear_model import Ridge     # Ridge regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
from sklearn import preprocessing

from copy import deepcopy
from sklearn.model_selection import train_test_split

def criteria(pre_value,true_value):
    pres = deepcopy(pre_value)
    for i in range(len(pres)):
        pres[i] = float(pres[i])
    pres = np.array(pres)

    trues = deepcopy(true_value)
    for i in range(len(trues)):
        trues[i] = float(trues[i])
    trues = np.array(trues)

    APE = abs(pres-trues)/trues
    MAPE = sum(APE)/len(APE)
    RMSE = math.sqrt((sum((pres-trues)**2))/len(pres))
    r2 = r2_score(true_value,pre_value)
    return MAPE, RMSE, r2


data = Ds.importxlsx() # File path

temp_X =[temp[0:5] for temp in data[1:]]
temp_Y =[temp[-1] for temp in data[1:]]
X=[]
Y=[]
for i in range(len(temp_X)):
    tempX=[]
    for j in temp_X[i]:
        tempX.append(float(j))
    X.append(tempX)
    Y.append(float(temp_Y[i]))
print(len(X))

regr = RandomForestRegressor(n_estimators=20000, min_samples_leaf=3, oob_score=True)
regr.fit(X, Y)
importances=regr.feature_importances_
print('###########随机森林重要性#######################')
for temp in importances:
    print(temp) # RandomForest importances
