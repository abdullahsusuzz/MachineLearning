# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 07:00:38 2018

@author: Abdullah
"""

#1.importlar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2.veri on işleme

#2.1.veri okuma
veriler = pd.read_csv('eksikveriler.csv')

#2.2imputer eksik veri doldurma
yas = veriler.iloc[:,1:4].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy = 'mean',axis=0)
yas[:,1:4] = imputer.fit_transform(yas[:,1:4])
        
#2.3 encoder veri tiplerini duzenleme 
#(direk sirasi ile sayi veren)
ulke = veriler.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])

#her veriyi sutunlra ayırarak kodlama
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray() 

#bir kolonun alınması
cinsiyet = veriler.iloc[:,-1:].values

#2.4 verileri dataframe turune donusturulmesi
sonuc = pd.DataFrame(data = ulke,index=range(22),columns=['fr','tr','us'])
sonuc2 = pd.DataFrame(data = yas,index=range(22),columns=['boy','kilo','yas'])
sonuc3 = pd.DataFrame(data = cinsiyet,index=range(22),columns=['cinsiyet'])
#2.5.dataframelerin birlestirme
s=pd.concat([sonuc,sonuc2],axis = 1)

#2.6.verileri egitim ve test olarak ayırma
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#♠2.7.veriyi standartlastırma
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc = StandardScaler()#standartaziation
mc = MinMaxScaler()#normalization
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)





