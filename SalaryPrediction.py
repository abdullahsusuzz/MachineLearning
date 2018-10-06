# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 22:10:08 2018

@author: abdullahsusuzz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score

veriler = pd.read_csv('maaslar_yeni.csv')
print(veriler.corr())
X = veriler.iloc[:,2:4].values
Y = veriler.iloc[:,-1:].values


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaler = scaler.fit_transform(X)
scaler2= StandardScaler()
y_scaler = scaler2.fit_transform(Y)


#metodun kendisi
#kernel =rbf
from sklearn.svm import SVR
svr_reg  = SVR(kernel='rbf')
svr_reg.fit(x_scaler,y_scaler)


from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)

print("lin_reg")
model=sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

print('poly_reg')
model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

print('svr_reg')
model3=sm.OLS(svr_reg.predict(x_scaler),x_scaler)
print(model3.fit().summary())

print('dt_reg')
model4=sm.OLS(dt_reg.predict(X),X)
print(model4.fit().summary())

print('rf_reg')
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())



print(lin_reg.predict(4,10,100))