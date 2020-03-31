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
from xgboost import XGBRegressor
# Importing the dataset
# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 0:10].values #taking indexes from 1st to 9th one barring defcon level and id
y = dataset.iloc[:, 10:11].values # penultimate column storing class

np.unique(y)

for i in range(10000):
    X[i][9]=X[i][9]*1000
    
    

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.1, random_state = 130) #test on 2000 observations and train on 8000



        

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 130) #test on 2000 observations and train on 8000



# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1));
X_new=sc.fit_transform(X);

#Random forest 

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100,max_depth=50,max_features=8, min_samples_leaf=3, min_samples_split=8);
print(model)
model.fit(X_train,y_train);
 #56
 
model=RandomForestClassifier(n_estimators=300,max_depth=200,max_features=9, min_samples_leaf=5, min_samples_split=12);
print(model)
model.fit(X_train,y_train);
 #55



#parameter tuning
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid ={'nthread':[None,4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05,0.1,0.4], #so called `eta` value
              'max_depth': [3,6],
              'min_child_weight': [1,5,11],
              'silent': [None,1],
              'subsample': [0.6,1,0.8],
              'colsample_bytree': [0.7,1],
              'n_estimators': [100,200,500], #number of trees, change it to 1000 for better results
              'missing':[None],
              'seed': [1337,None]}

model=XGBClassifier();
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_


#predictions
model.score(X_train,y_train)
model.score(X_test,y_test)

# Importing the test dataset
dataset_test = pd.read_csv('test.csv')
dataset_test["All_Host"]=dataset_test["Allied_Nations"]/dataset_test["Hostile_Nations"]
dataset_test["Act_Inact"]=dataset_test["Active_Threats"]/dataset_test["Inactive_Threats"]
dataset_test=dataset_test.drop(["Allied_Nations","Hostile_Nations","Active_Threats","Inactive_Threats"],axis=1)

X_final=dataset_test.iloc[:, np.r_[0:6,7:9]].values;
ids=dataset_test.iloc[:, 6:7].values;


model=XGBClassifier(colsample_bytree=0.8,
 learning_rate=0.2,
 max_depth=25,
 min_child_weight=4,
 missing=None,
 n_estimators=35,
 nthread=None,
 objective='binary:logistic',
 seed=None,
 silent=None,
 subsample=0.8,
 random_state=130);
                    
print(model)
model.fit(X_train,y_train);

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))




sev=[];
for i in range(2500):
    lt=[]
    for k in range(10):
        lt.append(X_final[i][k])
    ans=model.predict(sc.transform(np.array([lt])))
    
    sev.append((ids[i],ans[0]));
        
answers=[]


for i in range(2500):
    answers.append(sev[i][1]);
    
    

        
#print(model)
    


#hyper parameter tuning
    
from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(X_train,y_train);
#predictions
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))



    
