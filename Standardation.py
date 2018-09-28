# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:04:42 2018

@author: abdullahsusuzz

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veri = pd.read_csv('satislar.csv')
satislar = veri[['Satislar']]#eger tek koseli parantez kullanırssak seri tipinde olur
aylar = veri[['Aylar']]

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,
                                                 random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
                      #suan standartlasmamıs verş uzerinnden hesap yapıldı ama
                      #standartlasırsa daha iyi sonuc veriri o daha sonra
tahmin = lr.predict(x_test)

X_train = X_train.sort_index()
Y_train = Y_train.sort_index()

plt.plot(X_train,Y_train) 
plt.plot(X_test,lr.predict(X_test))


plt.title('Aylara gore urun satısı')
plt.xlabel('Aylar')
plt.ylabel('Urunler')