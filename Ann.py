# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 01:55:33 2021

@author: Loai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tf.__version__

# Importing the dataset
df = pd.read_csv('data important.csv')
X=df.iloc[:,2:12]
Y=df.iloc[:,1:2]
# print(df.info())
# df.drop(["id",'Unnamed: 32'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
label_features = ['diagnosis']
Y = df[label_features].apply(LabelEncoder().fit_transform)


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=.25, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=15, activation='relu',input_dim=10))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu',input_dim=15))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Training the ANN on the Training set
ann.fit(train_x, train_y, batch_size = 15, epochs = 100)

# # Predicting the Test set results
y_pred = ann.predict(test_x)
y_pred = (y_pred > 0.5)
# # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(train_y),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score

cm = confusion_matrix(test_y, y_pred)
f1 = f1_score(test_y, y_pred,average=None)
roc = roc_auc_score(test_y, y_pred)
acu=accuracy_score(test_y, y_pred)

