# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:12:32 2018

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

Z =X-0.4
K = X+0.4

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)
#random forest decision treeyi kullanır fakat elindeki veriyi alt parcalara 
#ayırarak birden fazla decision tree kullanır ve en sonunda hepsinin ortalamasını alır
#bu yuzden ezberledigi verinin haricindede sonuc donebilir
plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')
plt.plot(Z,rf_reg.predict(Z),color='green')
plt.plot(K,rf_reg.predict(K),color='yellow')

print(rf_reg.predict(8.5))