# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:59:55 2020

@author: Subham
"""

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
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 0:10].values #taking indexes from 1st to 9th one barring defcon level and id
y = dataset.iloc[:, 10:11].values # penultimate column storing class

np.unique(y)

for i in range(10000):
    X[i][9]=X[i][9]*1000
    

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1));
X_new=sc.fit_transform(X);

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.1, random_state = 130) #test on 2000 observations and train on 8000



rfc=RandomForestClassifier(bootstrap=True, max_depth=40, max_features=10, min_samples_leaf=5, min_samples_split=7, n_estimators=100,random_state=130)

xgb=XGBClassifier(colsample_bytree=0.8,
 learning_rate=0.2,
 max_depth=25,min_child_weight=4,missing=None,n_estimators=35,nthread=None,objective='binary:logistic',seed=None,silent=None,subsample=0.8,random_state=130);

rfc.fit(X_train,y_train);
xgb.fit(X_train,y_train);


#predictions
print(rfc.score(X_train,y_train))
print(rfc.score(X_test,y_test))
print(xgb.score(X_train,y_train))
print(xgb.score(X_test,y_test))

# Importing the test dataset
dataset_test = pd.read_csv('test.csv')

X_final=dataset_test.iloc[:, 0:10].values;
ids=dataset_test.iloc[:, 10:11].values;

for i in range(2500):
    X_final[i][9]=X_final[i][9]*1000



sev=[];
for i in range(2500):
    lt=[]
    for k in range(10):
        lt.append(X_final[i][k])
    ans=rfc.predict(sc.transform(np.array([lt])))
    
    sev.append((ids[i],ans[0]));
        
answers1=[]


for i in range(2500):
    answers1.append(sev[i][1]);
    
  
sev=[];
for i in range(2500):
    lt=[]
    for k in range(10):
        lt.append(X_final[i][k])
    ans=xgb.predict(sc.transform(np.array([lt])))
    
    sev.append((ids[i],ans[0]));
        
answers2=[]


for i in range(2500):
    answers2.append(sev[i][1]);

        
#print(model)
    
answers=[]
for i in range(2500):
    answers.append((answers1[i]+answers2[i])//2);

#min - 54
#max - 55
#avg - 54
    
    
    
    
# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[50,75,100],
              "learning_rate":  [0.3],
              "random_state": [130]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=3, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_new,y.ravel())

ada_best = gsadaDTC.best_estimator_



ada_best.fit(X_train,y_train);
print(ada_best.score(X_train,y_train))
print(ada_best.score(X_test,y_test))


print(ada_best.feature_importances_)
print(rfc.feature_importances_)
print(xgb.feature_importances_)




#voting

votingC = VotingClassifier(estimators=[('rfc', rfc), ('xgb',xgb), ('adac',ada_best)], voting='hard', n_jobs=4)

votingC = votingC.fit(X_train, y_train)

print(votingC.score(X_train,y_train))
print(votingC.score(X_test,y_test))



#predict using only adaboost

sev=[];
for i in range(2500):
    lt=[]
    for k in range(10):
        lt.append(X_final[i][k])
    ans=ada_best.predict(sc.transform(np.array([lt])))
    
    sev.append((ids[i],ans[0]));
        
answers3=[]


for i in range(2500):
    answers3.append(sev[i][1]);


#predict using ensemble

sev=[];
for i in range(2500):
    lt=[]
    for k in range(10):
        lt.append(X_final[i][k])
    ans=votingC.predict(sc.transform(np.array([lt])))
    
    sev.append((ids[i],ans[0]));
        
answers4=[]


for i in range(2500):
    answers4.append(sev[i][1]);