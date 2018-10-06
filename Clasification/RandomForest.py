# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 19:22:50 2018

@author: abdullahsusuzz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('veriler.csv')

X = veriler.iloc[5:,1:4].values
Y = veriler.iloc[5:,4:].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(criterion = 'gini')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

cm=confusion_matrix(y_pred,y_test)
print(cm)