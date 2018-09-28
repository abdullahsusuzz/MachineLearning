# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 22:19:06 2018

@author: Abdullah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')

bky = veriler.iloc[:,1:4].values
cinsiyet = veriler.iloc[:,-1:].values
ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray() 

cinsiyet[:,-1] = le.fit_transform(cinsiyet[:,-1])
cinsiyet = ohe.fit_transform(cinsiyet).toarray()

sonuc = pd.DataFrame(data = ulke,index=range(22),columns=['fr','tr','us'])
sonuc2 = pd.DataFrame(data = bky,index=range(22),columns=['boy','kilo','yas'])
sonuc3 = pd.DataFrame(data = cinsiyet[:,1:],index=range(22),columns=['cinsiyet'])

s = pd.concat([sonuc,sonuc2],axis=1)
s2 = pd.concat([s,sonuc3],axis=1)


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size = 0.33,random_state=0)


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)

y_preg = regression.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)

Boy = pd.DataFrame(data=boy,index=range(22),columns=['boy'])

ilkpart = s2.iloc[:,0:3].values
sonpart = s2.iloc[:,4:].values

Ilkpart = pd.DataFrame(data=ilkpart,index=range(22),columns=['fr','tr','us'])
Sonpart = pd.DataFrame(data=sonpart,index=range(22),columns=['kilo','yas','cinsiyet'])

veri =pd.concat([Ilkpart,Sonpart],axis=1)

x_train,x_test,y_train,y_test = train_test_split(veri,Boy,test_size = 0.33,
                                                 random_state=0)


regression.fit(x_train,y_train)

y_preg = regression.predict(x_test)

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)
X_l= veri.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog=Boy,exog=X_l)
r = r_ols.fit()
print(r.summary())

X_l= veri.iloc[:,[0,1,2,3,5]].values
r_ols = sm.OLS(endog=Boy,exog=X_l)
r = r_ols.fit()
print(r.summary())

X_l= veri.iloc[:,[0,1,2,3]].values
r_ols = sm.OLS(endog=Boy,exog=X_l)
r = r_ols.fit()
print(r.summary())
