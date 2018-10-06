# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:59:36 2018

@author: abdullahsusuzz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')

X = veriler.iloc[5:,1:4].values
Y = veriler.iloc[5:,4:].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train)

y_preg = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_preg,y_test)
print(cm)
