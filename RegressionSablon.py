# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:19:36 2018

@author: abdullahsusuzz
"""
#kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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

#support vector regression
#bu metot kullanmadan once degerler scale edilir
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

#kernel = linear
svr_reg2  = SVR(kernel='linear')
svr_reg2.fit(x_scaler,y_scaler)

#kernel = poly
svr_reg3  = SVR(kernel='poly')
svr_reg3.fit(x_scaler,y_scaler)

#decision tree regression
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)

#random forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)


#gorsellestirme
plt.title('linear regression')
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

plt.title('2.derece polynomial regression')
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)),color='blue')
plt.show()

plt.title('4.derece polynomial regression')
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color='blue')
plt.show()

plt.title('support vector machine regression rbf')
plt.scatter(x_scaler,y_scaler,color='red')
plt.plot(x_scaler,svr_reg.predict(x_scaler),color='blue')
plt.show()

plt.title('support vector machine regression linear')
plt.scatter(x_scaler,y_scaler,color='red')
plt.plot(x_scaler,svr_reg2.predict(x_scaler),color='blue')
plt.show()

plt.title('support vector machine regression poly')
plt.scatter(x_scaler,y_scaler,color='red')
plt.plot(x_scaler,svr_reg3.predict(x_scaler),color='blue')
plt.show()

plt.title('decision tree regression')
plt.scatter(X,Y,color='red')
plt.plot(X,dt_reg.predict(X),color='blue')
plt.show()

plt.title('random forest regression')
plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')
plt.show()

print("------------------------------------")
print("linear regression :")
print(r2_score(Y,lin_reg.predict(X)))

print("2.derece Polynomial regression :")
print(r2_score(Y,lin_reg2.predict(poly_reg2.fit_transform(X))))

print("4.derece polynomial regression :")
print(r2_score(Y,lin_reg3.predict(poly_reg3.fit_transform(X))))

print("supoort vector regression rbf :")
print(r2_score(y_scaler,svr_reg.predict(x_scaler)))

print("supoort vector regression linear:")
print(r2_score(y_scaler,svr_reg2.predict(x_scaler)))

print("supoort vector regression poly :")
print(r2_score(y_scaler,svr_reg3.predict(x_scaler)))

print("decision tree regression :")
print(r2_score(Y,dt_reg.predict(X)))

print("random forest regression :")
print(r2_score(Y,rf_reg.predict(X)))

#polinom regresyonun lineardan kunllanım farkı sadece alacagı degeri polinom
#donusumu yapılarak verilmesidir
#dısardan alınan bir verinin tahmin ettirilmesi
#print(lin_reg.predict(11))#linear regression
#print(lin_reg2.predict(poly_reg2.fit_transform(11)))#polinom regression
 