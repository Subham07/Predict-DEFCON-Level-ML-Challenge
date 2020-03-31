# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:29:45 2020

@author: Subham
"""



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils import np_utils
from xgboost import XGBClassifier


# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 0:10].values #taking indexes from 1st to 9th one barring defcon level and id
y = dataset.iloc[:, 10:11].values # penultimate column storing class

np.unique(y)


#visualization
import matplotlib.pyplot as plt
colors={1:'r',2:'g',3:'b',4:'y',5:'k'}

xp=[]
yp=[]
for i in range(10000):
    xp.append(X[i][0])
    yp.append(X[i][7])


fig, ax = plt.subplots()
for i in range(10000):
    ax.scatter(xp[i],yp[i],color=colors[y[i][0]])

ax.set_title('DEFCON')
ax.set_xlabel('closest threat distance')
ax.set_ylabel('troops mobilized')


# Importing the test dataset
dataset_test = pd.read_csv('test.csv')

X_final=dataset_test.iloc[:, 0:10].values;
ids=dataset_test.iloc[:, 10:11].values;
