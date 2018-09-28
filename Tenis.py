# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 19:58:20 2018

@author: Abdullah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

veri = pd.read_csv('odev_tenis.csv')
veriler2 = veri.apply(LabelEncoder().fit_transform)

outlook = veri.iloc[:,0:1].values
temparature = veri.iloc[:,[1]].values
humidity = veri.iloc[:,[2]].values
windy = veriler2.iloc[:,3:].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
outlook[:,0] = le.fit_transform(outlook[:,0])


windy[:,0] = le.fit_transform(windy[:,0])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()



weather = pd.DataFrame(data = outlook,index=range(14),columns=['overcast','rainy','sunny'])
tempra = pd.DataFrame(data = temparature,index=range(14),columns=['temprature'])
humit = pd.DataFrame(data = humidity,index=range(14),columns=['humidity'])
sonveriler = pd.DataFrame(data = windy,index=range(14),columns=['windy','play'])

s = pd.concat([weather,humit],axis=1)
s2 = pd.concat([s,sonveriler],axis=1)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s2,tempra,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
li = LinearRegression()
li.fit(x_train,y_train)
y_preg = li.predict(x_test)

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=s2,axis=1)
X_l= s2.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog=tempra,exog=X_l)
r = r_ols.fit()
print(r.summary())

x_train,x_test,y_train,y_test=train_test_split(weather,tempra,test_size=0.33,random_state=0)

li.fit(x_train,y_train)
y2_preg = li.predict(x_test)

















