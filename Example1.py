# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 06:54:20 2018

@author: abdullahsusuzz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

le = LabelEncoder()
ohe = OneHotEncoder()
regression = LinearRegression()

veriler = pd.read_csv('odev_tenis.csv')
veriler2 = veriler.apply(LabelEncoder().fit_transform)
outlook = veriler2.iloc[:,0:1].values

outlooks = ohe.fit_transform(outlook).toarray()
tempra = veriler.iloc[:,1:2].values
others = veriler2.iloc[:,3:].values

humidity = veriler.iloc[:,2:3].values

dataFrameOutlook = pd.DataFrame(data=outlooks,index=range(14),columns=['o','r','s'])
dataFrameTemprature = pd.DataFrame(data=tempra,index=range(14),columns=['temprature'])
dataFrameOthers = pd.DataFrame(data=others,index=range(14),columns=['windy','play'])

trainData = pd.concat([dataFrameOthers,dataFrameTemprature],axis=1)
trainData = pd.concat([trainData,dataFrameOutlook],axis=1)

resultData = pd.DataFrame(data = humidity,index = range(14),columns=['humit'])

x_train,x_test,y_train,y_test = train_test_split(trainData,resultData,test_size=0.33,random_state=0)


regression.fit(x_train,y_train)
y_preg = regression.predict(x_test)

X = np.append(arr =np.ones((14,1)).astype(int),values = trainData, axis=1)
X_l= trainData.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog =resultData,exog=X_l)
r = r_ols.fit()
print(r.summary())

print(y_preg)

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regression.fit(x_train,y_train)
y_preg2 = regression.predict(x_test)

X = np.append(arr =np.ones((14,1)).astype(int),values = trainData, axis=1)
X_l= trainData.iloc[:,[1,2,3,4,5]].values
r_ols = sm.OLS(endog =resultData,exog=X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,-1:]
x_test = x_test.iloc[:,-1:]

regression.fit(x_train,y_train)
y_preg3 = regression.predict(x_test)

X = np.append(arr =np.ones((14,1)).astype(int),values = trainData, axis=1)
X_l= trainData.iloc[:,[1,2,3,4]].values
r_ols = sm.OLS(endog =resultData,exog=X_l)
r = r_ols.fit()
print(r.summary())
