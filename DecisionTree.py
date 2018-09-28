# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 22:39:45 2018

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
#bu regresyonun sıkıntısı ilk olarak olusan sonuclar haricindeki tahminlerin 
#sonuclarınıda hep aynı bulması
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,dt_reg.predict(X),color='blue')

print(dt_reg.predict(10.4))#50000
print(dt_reg.predict(9.6))#50000
#iki degeride aynı yere map ettiginden iki sonucta aynı cıkar