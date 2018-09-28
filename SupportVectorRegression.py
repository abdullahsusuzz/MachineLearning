# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:27:08 2018

@author: abdullahsusuzz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

X = x.values
Y = y.values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaler = scaler.fit_transform(X)
scaler2 = StandardScaler()
y_scaler = scaler2.fit_transform(Y)

from sklearn.svm import SVR

#rbf
svr_reg1 = SVR(kernel ='rbf')
svr_reg1.fit(x_scaler,y_scaler)

#lineaar
svr_reg2 = SVR(kernel ='linear')
svr_reg2.fit(x_scaler,y_scaler)

#poly
svr_reg2= SVR(kernel ='poly')
svr_reg2.fit(x_scaler,y_scaler)

plt.title('rbf')
plt.scatter(x_scaler,y_scaler)
plt.plot(x_scaler,svr_reg1.predict(x_scaler))
plt.show()

plt.title('linear')
plt.scatter(x_scaler,y_scaler)
plt.plot(x_scaler,svr_reg2.predict(x_scaler))
plt.show()

plt.title('poly')
plt.scatter(x_scaler,y_scaler)
plt.plot(x_scaler,svr_reg2.predict(x_scaler))
plt.show()

