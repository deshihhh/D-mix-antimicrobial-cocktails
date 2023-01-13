# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:18:03 2021

@author: Lenovo
"""
from tqdm import tqdm
import numpy as np  # numpy库
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor # 决策树回归
from sklearn.ensemble import RandomForestRegressor # 随机森林回归
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sci
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.utils import resample

# 数据准备
data = pd.read_excel("C:\\Users\YJZ\Desktop\\YHYJZ.xlsx")
x1, y1 = data.iloc[:,0:5], data.iloc[:,5]

llist = []
for i in range(0,101,5):#A
    for j in range(0,101,5):#B
        for k in range(0,101,5):#C
            for l in range(0,101,5):#D
                for m in range(0,101,5):#E
                    if i+j+k+l+m == 100:
                        data = [i,j,k,l,m]
                        llist.append(data)
space = np.array(llist)
spacelen = len(space)
print('成分个数：%d' %spacelen) #10626

# 回归模型
def Model(k1, data_predict):
    param_grid = [{'n_estimators': [10,100],'min_samples_split': [2,5,10,20],'min_samples_leaf': [1,5,10]}]
    model = RandomForestRegressor(random_state=0)
    grid = GridSearchCV(model, param_grid, cv=10,scoring='neg_mean_squared_error')
    grid.fit(k1[:,:-1],k1[:,-1])
    rf1 = grid.best_estimator_
    model = rf1.fit(k1[:,:-1],k1[:,-1])
    predictions = model.predict(data_predict) 
    return(predictions)
    print(preditions)

predict_dataa= np.arange(0,spacelen)
t1 = x1.shape[0]
q = 10
for u in range(0,q): 
    print(u)
    X11 , y11 = resample(x1, y1, replace=True, n_samples = t1, random_state= u)
    k11 = np.column_stack((X11, y11))
    predict_data = Model(k11, space)
    predict_dataa = np.column_stack((predict_dataa, predict_data)) 
data_predict = np.column_stack((np.mean(predict_dataa[:,1:,],axis=1), np.std(predict_dataa[:,1:,],axis=1)))
kk = np.column_stack((space,data_predict))

def cal_ei(mydata, searchspace0):
    ego = (min(mydata[:])-searchspace0[:,-2])/(searchspace0[:,-1])
    ei_ego = searchspace0[:,-1]*ego*sci.stats.norm.cdf(ego) + searchspace0[:,-1]*sci.stats.norm.pdf(ego)  #    kg = (searchspace0[:,8] - max(max(searchspace0[:,8]),max(mydata[:,-1])))/(searchspace0[:,9]) #    ei_kg = searchspace0[:,9]*kg*stats.norm.cdf(kg) + searchspace0[:,9]*stats.norm.pdf(kg)    #    max_P = stats.norm.cdf(ego, loc=searchspace0[:,8], scale=searchspace0[:,9])
    ei =np.column_stack((searchspace0, ei_ego))
    return (ei)

EI = cal_ei(y1, kk)
aEI1 = np.array(EI)
aei = pd.DataFrame(aEI1)
print(aEI1.shape[0])
aei.to_csv('prd.csv')













