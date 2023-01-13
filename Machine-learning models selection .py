# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:05:52 2021

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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
# 数据准备
data = pd.read_excel("C:\\Users\YJZ\Desktop\\YHYJZ.xlsx")
x1, y1 = data.iloc[:,1:-1], data.iloc[:,-1]
# 训练回归模型
##### 1、 支持向量机回归
k_range = np.arange(100,1001,100)
k2_range = np.arange(0.01,0.11,0.01)
param_grid_svr = dict(C = k_range, gamma = k2_range)
model_svr = SVR(kernel='rbf')
##### 2、线性回归
param_grid_lr = [{}]
model_lr = LinearRegression()
##### 3、 自适应集成回归
param_grid_abr = [{'learning_rate': [0.1,0.5,1,10], 'n_estimators': [10,100]}] 
model_abr = AdaBoostRegressor(random_state=0)  # 建立自适应增强回归模型对象
##### 4、 近邻回归
param_grid_knr = [{'n_neighbors': [1,2,3,4,5],'leaf_size': [1,5,10,20,30,40,50,60,70,80,90,100]}]
model_knr = neighbors.KNeighborsRegressor(weights='distance')  # k近邻回归
##### 5、 决策树回归
param_grid_dtr = [{'min_samples_split': [2,3,4,5,6,7,8,9],'min_samples_leaf': [1,5,10,20,30,40,50,60,70,80,90,100]}]
model_dtr = DecisionTreeRegressor(max_depth=7,random_state=0) # 决策树回归
#### 6、 随机森林回归
param_grid_rfr = [{'n_estimators': [10,100],'min_samples_split': [2,5,10,20],'min_samples_leaf': [1,5,10]}]
model_rfr = RandomForestRegressor(random_state=0)
#### 7、 梯度提升回归
param_grid_gbr = [{'learning_rate': [0.1,0.5,1,10], 'n_estimators': [10,100], 'max_depth': [1,3,5,10,20],
                   'min_samples_split': [2,5,10,20], 'min_samples_leaf': [1,5,10]}]
##### 8、 高斯过程回归
param_grid_gpr = [{'alpha': [1e-11,1e-10,1e-9], 'n_restarts_optimizer': [5,10,20,50]}]
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
model_gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
#### 9、 多项式回归
# param_grid_poly = [{}]
# poly = PolynomialFeatures(2)
# x1=poly.fit_transform(x1)
# x2=poly.fit_transform(x2)
# x3=poly.fit_transform(x3)
# x4=poly.fit_transform(x4)
# model_poly = LinearRegression()
##### 10、Xgboost回归
# param_grid_Xg = [{'learning_rate': [0.1,0.5,1,10], 'n_estimators': [10,100], 'max_depth': [1,3,5,10,20]}]
# model_Xg = xgb.XGBRegressor(random_state=0)
##### 11、人工神经网络回归
param_grid_ANNmlp = [{'alpha': [0.0001,0.001,0.01], 'learning_rate_init': [0.001,0.01,0.1],
                      'max_iter': [5000],'tol': [0.001,0.01]}] 
model_ANNmlp = MLPRegressor(random_state=0)

# model_names = ['XGR']
# param_grid = [param_grid_Xg]
# model_dic = [model_Xg]

model_names = ['ANN']
param_grid = [param_grid_ANNmlp]
model_dic = [model_ANNmlp]

# model_names = ['GPR']
# param_grid = [param_grid_gpr]
# model_dic = [model_gpr]

# model_names = ['LR','SVR','DTR','KNR','RF','GBR','ABR','GPR','XGR']  # 'LR','SVR','DTR','KNR','RF','GBR','ABR','GPR','XGR'
# param_grid = [param_grid_lr, param_grid_svr, param_grid_dtr, param_grid_knr, param_grid_rfr, param_grid_gbr, param_grid_abr, param_grid_gpr, param_grid_Xg]  # param_grid_lr, param_grid_svr, param_grid_dtr, param_grid_knr, param_grid_rfr, param_grid_gbr, param_grid_abr, param_grid_gpr, param_grid_Xg
# model_dic = [model_lr, model_svr, model_dtr, model_knr, model_rfr, model_gbr, model_abr, model_gpr, model_Xg]  # model_lr, model_svr, model_dtr, model_knr, model_rfr, model_gbr, model_abr, model_gpr, model_Xg

# model_names = ['poly']  # 'poly'
# param_grid = [param_grid_poly]  #param_grid_poly
# model_dic = [model_poly]  # model_poly

y_plot = np.arange(2)
tmp_list = []
# temp = []
z = 0
for model, param in zip(model_dic, param_grid):  # 读出每个回归模型对象
    r2_train, r2_test, mse_train, mse_test = np.arange(1),np.arange(1),np.arange(1),np.arange(1)
    cvs = np.arange(1)
    coef = np.arange(9)
    cintercept = np.arange(1)
    temp = []
    for f in tqdm(range(100), unit='次', desc='time'):
        X_train,X_test,y_train,y_test=train_test_split(x1, y1, test_size=0.2, random_state=f)
        grid = GridSearchCV(model, param, cv=10,scoring='neg_mean_squared_error')  #neg_mean_squared_error
        grid.fit(X_train, y_train)
        grid_est = grid.best_estimator_
        grid_par = grid.best_params_
        modell = grid_est.fit(X_train, y_train)
        
        aa = cross_val_score(modell, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        cvs = np.row_stack((cvs, np.mean(aa)))
        
        # 线性模型拟合权重参数
        # parameters1 = modell.coef_
        # parameters2 = modell.intercept_
        # coef = np.column_stack((coef, parameters1))
        # cintercept = np.column_stack((cintercept, parameters2))

        # scores_train = cross_val_score(modell, X_train, y_train,cv=5,scoring='neg_mean_squared_error') #训练集交叉验证
        y_pred = modell.predict(X_test)
        y_train_pred = modell.predict(X_train)
        
        r2__train = r2_score(y_train,y_train_pred)        
        r2_train = np.row_stack((r2_train, r2__train))
        mse__train = mean_squared_error(y_train,y_train_pred)
        mse_train = np.row_stack((mse_train, mse__train))

        r2__test = r2_score(y_test, y_pred)
        r2_test = np.row_stack((r2_test, r2__test))
        mse__test = mean_squared_error(y_test, y_pred)
        mse_test = np.row_stack((mse_test, mse__test))

        yplot = np.column_stack((y_test, y_pred))
        y_plot = np.row_stack((y_plot, yplot))

    trainMSEmean = np.mean(mse_train[1:,:])
    trainMSEstd = np.std(mse_train[1:,:])
    trainR2mean = np.mean(r2_train[1:,:])
    trainR2std = np.std(r2_train[1:,:])

    testMSEmean = np.mean(mse_test[1:,:])
    testMSEstd = np.std(mse_test[1:,:])
    testR2mean = np.mean(r2_test[1:,:])
    testR2std = np.std(r2_test[1:,:])

    temp.append(trainMSEmean)
    temp.append(trainMSEstd)
    temp.append(trainR2mean)
    temp.append(trainR2std)
    
    temp.append(testMSEmean)
    temp.append(testMSEstd)
    temp.append(testR2mean)
    temp.append(testR2std)

    temp = np.array(temp)
    tmp_list.append(temp)
    cvss = np.mean(cvs[1:,:])
    del temp
    del r2__train
    del r2__test
    del mse__train
    del mse__test
    z += 1
    print (70 * '-')  # 打印分隔线
    print('-- Mission %d Complete --' %z )

# 模型效果指标评估
# df1 = pd.DataFrame(cv_score_list,index=model_names)  # 建立交叉检验的数据框
df2 = pd.DataFrame(tmp_list, index=model_names, columns=['trainMSEmean','trainMSEstd','trainR2mean','trainR2std','testMSEmean','testMSEstd','testR2mean','testR2std'])  # 建立回归指标的数据框
#print (70 * '-')  # 打印分隔线
#print ('cross validation result:')  # 打印输出标题
#print (df1)  # 打印输出交叉检验的数据框
print (70 * '-')  # 打印分隔线
print ('regression metrics:')  # 打印输出标题
print (df2)  # 打印输出回归指标的数据框
print (70 * '-')  # 打印分隔线

