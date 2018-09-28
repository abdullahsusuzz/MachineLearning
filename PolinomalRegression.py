# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:19:36 2018

@author: abdullahsusuzz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataSet = pd.read_csv('maaslar.csv')

x = dataSet.iloc[:,1:2]
y = dataSet.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

poly_reg = PolynomialFeatures(degree=5)
x_poly=poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

#polinom regresyonun lineardan kunllanım farkı sadece alacagı degeri polinom
#donusumu yapılarak verilmesidir

print(lin_reg.predict(11))#linear regression
print(lin_reg2.predict(poly_reg.fit_transform(11)))#polinom regression
 