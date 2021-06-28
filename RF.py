# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 02:10:55 2021

@author: Loai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('data important.csv')
X=df.iloc[:,2:12]
Y=df.iloc[:,1:2]

from sklearn.preprocessing import LabelEncoder
label_features = ['diagnosis']
Y = df[label_features].apply(LabelEncoder().fit_transform)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=.25, random_state=0)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifier.fit(train_x, train_y.values.ravel())

# Predicting a new result
y_pred= classifier.predict(test_x)

# Making the Confusion Matrix
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score

cm = confusion_matrix(test_y, y_pred)
f1 = f1_score(test_y, y_pred,average=None)
roc = roc_auc_score(test_y, y_pred)
acu=accuracy_score(test_y, y_pred)

