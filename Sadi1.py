# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:05:20 2018

@author: Abdullah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri okuma import
veriler = pd.read_csv('veriler.csv')
print(veriler)
#istenilen kolonu alma
boy = veriler[['boy']]
print(boy)
#eksik verileri doldurma
eksikVeriler = pd.read_csv('eksikveriler.csv')
print(eksikVeriler)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy='mean',axis=0)

Yas = eksikVeriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)

ulke = eksikVeriler.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

sonuc = pd.DataFrame(data = ulke,index = range(22),columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22),columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = eksikVeriler.iloc[:,-1:].values
print(cinsiyet)
sonuc3 = pd.DataFrame(data = cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)
s = pd.concat([sonuc,sonuc2],axis=1)
s1 = pd.concat([s,sonuc3],axis=1)
print(s1)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)











