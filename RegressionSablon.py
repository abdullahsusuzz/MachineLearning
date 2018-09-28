# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:19:36 2018

@author: abdullahsusuzz
"""
#kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#veri import
dataSet = pd.read_csv('maaslar.csv')

#data frame bolumleme
x = dataSet.iloc[:,1:2]
y = dataSet.iloc[:,2:]

#dataset turundeki verileri numpy arraye donusturme
X = x.values
Y = y.values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#2.dereceden polinom regression modeli
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree=2)
x_poly2=poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2,Y)

#4.derecedenpolinom regression modeli
poly_reg3 = PolynomialFeatures(degree=5)
x_poly3=poly_reg3.fit_transform(X)
lin_reg3= LinearRegression()
lin_reg3.fit(x_poly3,Y)

#gorsellestirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)),color='blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color='blue')
plt.show()


#polinom regresyonun lineardan kunllanım farkı sadece alacagı degeri polinom
#donusumu yapılarak verilmesidir
#dısardan alınan bir verinin tahmin ettirilmesi
print(lin_reg.predict(11))#linear regression
print(lin_reg2.predict(poly_reg2.fit_transform(11)))#polinom regression
 