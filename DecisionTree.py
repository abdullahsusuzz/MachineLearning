# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 22:39:45 2021

@author: abdullahsusuzz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('play_tennis.csv')


X = veriler.iloc[:,1:5].values
Y = veriler.iloc[:,5:].values 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X_train = le.fit_transform(x_train[:,0])
X_test = le.transform(y_train[:,0])

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train,y_train)

y_pred= dtc.predict(X_test)

cm=confusion_matrix(y_pred,y_test)
print(cm)
