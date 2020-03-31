# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:36:00 2020

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
dataset=dataset.drop(["Diplomatic_Meetings_Set"],axis=1)
dataset=dataset.drop(["Allied_Nations"],axis=1)
dataset=dataset.drop(["Aircraft_Carriers_Responding"],axis=1)
dataset=dataset.drop(["Hostile_Nations"],axis=1)
X = dataset.iloc[:, 0:6].values #taking indexes from 1st to 9th one barring defcon level and id
y = dataset.iloc[:, 6:7].values # penultimate column storing class



np.unique(y)


# Importing the test dataset
dataset_test = pd.read_csv('test.csv')
dataset_test=dataset_test.drop(["Diplomatic_Meetings_Set"],axis=1)
dataset_test=dataset_test.drop(["Allied_Nations"],axis=1)
dataset_test=dataset_test.drop(["Aircraft_Carriers_Responding"],axis=1)
dataset_test=dataset_test.drop(["Hostile_Nations"],axis=1)
X_final=dataset_test.iloc[:, 0:6].values;
ids=dataset_test.iloc[:, 6:7].values;


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 130) #test on 2000 observations and train on 8000



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=300,max_depth=100,max_features=6, min_samples_leaf=3, min_samples_split=5);
print(model)
model.fit(X_train,y_train);
 #56
print("train set: ",model.score(X_train,y_train))
print("test set: ",model.score(X_test,y_test))   


sev=[];
for i in range(2500):
    lt=[]
    for k in range(6):
        lt.append(X_final[i][k])
    ans=model.predict(sc.transform(np.array([lt])))
    
    sev.append((ids[i],ans[0]));
        
answers=[]


for i in range(2500):
    answers.append(sev[i][1]);

# correlation heatmap
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

plt.figure(figsize=(6,4))
myBasicCorr = dataset.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(myBasicCorr,cmap=cmap, vmax=.9,vmin=-0.9,
     center=0, square=True, linewidths=.5)

plt.show()

citizen_fear=dataset["Citizen_Fear_Index"].values
plt.hist(citizen_fear)
plt.show()

closest_threat=dataset["Closest_Threat_Distance(km)"].values
plt.hist(closest_threat)
plt.show()


#dropping another column
dataset_test=dataset_test.drop(["Diplomatic_Meetings_Set"],axis=1)
dataset=dataset.drop(["Diplomatic_Meetings_Set"],axis=1)
X = dataset.iloc[:, 0:9].values #taking indexes from 1st to 9th one barring defcon level and id
y = dataset.iloc[:, 9:10].values # penultimate column storing class

np.unique(y)

X_final=dataset_test.iloc[:, 0:9].values;
ids=dataset_test.iloc[:, 9:10].values;



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 130) #test on 2000 observations and train on 8000



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=300,max_depth=100,max_features=9, min_samples_leaf=3, min_samples_split=8);
print(model)
model.fit(X_train,y_train);
 #56
print("train set: ",model.score(X_train,y_train))
print("test set: ",model.score(X_test,y_test))   


sev=[];
for i in range(2500):
    lt=[]
    for k in range(9):
        lt.append(X_final[i][k])
    ans=model.predict(sc.transform(np.array([lt])))
    
    sev.append((ids[i],ans[0]));
        
answers=[]


for i in range(2500):
    answers.append(sev[i][1]);
    
    
    
#dropping only allied nations > dropping only diplomatic meetings > dropping both