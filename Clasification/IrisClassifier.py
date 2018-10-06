# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 19:18:51 2018

@author: abdullahsusuzz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

veriler = pd.read_excel('Iris.xls')

X = veriler.iloc[:,:4].values
Y = veriler.iloc[:,4:].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y ,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0, solver='lbfgs')
logr.fit(X_train,y_train)

y_preg = logr.predict(X_test)
print("LogisticRegression")
cm = confusion_matrix(y_test,y_preg)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5,metric='minkowski')
knn.fit(X_train,y_train)

y_preg = knn.predict(X_test)
print("knn")
cm = confusion_matrix(y_test,y_preg)
print(cm)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_preg = gnb.predict(X_test)
print("naive bayes")
cm = confusion_matrix(y_test,y_preg)
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='gini')
dtc.fit(X_train,y_train)

y_preg = dtc.predict(X_test)
print("decision tree")
cm = confusion_matrix(y_test,y_preg)
print(cm)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)

y_preg = rfc.predict(X_test)
print("randomforest")
cm = confusion_matrix(y_test,y_preg)
print(cm)
