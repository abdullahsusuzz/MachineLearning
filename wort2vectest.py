# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:27:34 2018

@author: Abdullah
"""

import pandas as pd

veriler=pd.read_csv('testword2vec.csv')
veri = veriler.iloc[0:2,:].values
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
print(vectorizer)

corpus=['This is the first document',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?'
        ]

print(type(corpus))
X = vectorizer.fit_transform(corpus)
print(type(X))
print(veriler)
print(X.toarray())
from numpy import array
a = array( X )
#v = pd.DataFrame(data =a,index=range(3),columns=['a','b','c','d','e','f','g','h','i'])
print(a)