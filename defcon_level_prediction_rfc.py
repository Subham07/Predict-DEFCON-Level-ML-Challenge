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
from sklearn.ensemble import RandomForestClassifier

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
model=RandomForestClassifier(n_estimators=300,max_depth=100,max_features=10, min_samples_leaf=3, min_samples_split=8);
print(model)
model.fit(X_train,y_train);

#predictions
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
 #56




# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1));
X_new=sc.fit_transform(X);

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=300,max_depth=100,max_features=10, min_samples_leaf=3, min_samples_split=8);
print(model)
model.fit(X_train,y_train);

#predictions
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
 #56
 
 

model=RandomForestClassifier(n_estimators=300,max_depth=200,max_features=9, min_samples_leaf=5, min_samples_split=12);
print(model)
model.fit(X_new,y_train);
 #55

model=RandomForestClassifier(bootstrap=True, max_depth=40, max_features=10, min_samples_leaf=5, min_samples_split=7, n_estimators=100,random_state=130)
model.fit(X_train,y_train);
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
 #56

#parameter tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [20,30,40,50,60,70,80,90,100],
    'max_features': [7,8,9,10],
    'min_samples_leaf': [2,3,5,7,9],
    'min_samples_split': [2,3,5,7,9],
    'n_estimators': [20,30,40,50,60,70,80,90,100] 
}

model=RandomForestClassifier()
gsGBC = GridSearchCV(model,param_grid = param_grid, cv=3, scoring="accuracy", n_jobs= -1, verbose = 1)
gsGBC.fit(X_new,y);

print(gsGBC.best_score_)
print(gsGBC.best_params_)

model=RandomForestClassifier(n_estimators=300,max_depth=100,max_features=12, min_samples_leaf=3, min_samples_split=8);

# Instantiate the grid search model
all_accuracies=cross_val_score(estimator = model, X=X,y=y, 
                          cv = 10)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_


#predictions
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

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
    ans=model.predict(sc.transform(np.array([lt])))
    
    sev.append((ids[i],ans[0]));
        
answers=[]


for i in range(2500):
    answers.append(sev[i][1]);

#print(model)
    
    
#without scaling
    
sev=[];
for i in range(2500):
    lt=[]
    for k in range(10):
        lt.append(X_final[i][k])
    ans=model.predict(np.array([lt]))
    
    sev.append((ids[i],ans[0]));
        
answers=[]


for i in range(2500):
    answers.append(sev[i][1]);


#hyper parameter tuning
    
#parameter tuning
    
#using GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

ms=[25,50,75,100,125,150]
ne=[25,50,75,100,125,150]
for lr in ms:
    for mr in ne:
        model=GradientBoostingClassifier(learning_rate=0.1, loss='deviance', max_depth=8, max_features=6, min_samples_leaf=lr, n_estimators=mr)
        model.fit(X_train,y_train);
        #predictions
        print(lr,mr)
        print("train set: ",model.score(X_train,y_train))
        print("test set: ",model.score(X_test,y_test))


model=GradientBoostingClassifier(learning_rate=0.1, loss='deviance', max_depth=8, max_features=6, min_samples_leaf=25, n_estimators=75)
model.fit(X_train,y_train);
#56.74




#predictions



gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [50,100],
              'learning_rate': [0.1, 0.2],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }



gsGBC = GridSearchCV(model,param_grid = gb_param_grid, cv=3, scoring="accuracy", n_jobs= -1, verbose = 1)
gsGBC.fit(X_train,y_train);

print(gsGBC.best_score_)
print(gsGBC.best_params_)



#Extra trees classifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
#ExtraTrees 
ExtC = ExtraTreesClassifier(bootstrap=False,
 criterion='gini',
 max_depth=None,
 max_features=10,
 min_samples_leaf=1,
 min_samples_split=10,
 n_estimators=500)


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[400,500],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=3, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,y_train)

gsExtC.best_params_

model = ExtraTreesClassifier(bootstrap=False,
 criterion='gini',
 max_depth=None,
 max_features=10,
 min_samples_leaf=1,
 min_samples_split=10,
 n_estimators=50)
model.fit(X_train,y_train)
print("train set: ",model.score(X_train,y_train))
print("test set: ",model.score(X_test,y_test))




#voting classifier

model1=RandomForestClassifier(n_estimators=300,max_depth=200,max_features=9, min_samples_leaf=5, min_samples_split=12);
model2=GradientBoostingClassifier(learning_rate=0.1, loss='deviance', max_depth=8, max_features=6, min_samples_leaf=25, n_estimators=75)
model3 = ExtraTreesClassifier(bootstrap=False,criterion='gini',max_depth=None,max_features=10,min_samples_leaf=1,min_samples_split=10,n_estimators=50)

'''
model4=XGBClassifier(colsample_bytree=1,
 learning_rate=0.1,
 max_depth=6,
 min_child_weight=5,
 missing=None,
 n_estimators=50,
 nthread=None,
 objective='binary:logistic',
 seed=None,
 silent=None,
 subsample=0.8);
'''

from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[('gbc',model2),('rfc',model1),('etc',model3)], voting='soft', n_jobs=4)
model.fit(X_train,y_train)

print("train set: ",model.score(X_train,y_train))
print("test set: ",model.score(X_test,y_test))

sev=[];
for i in range(2500):
    lt=[]
    for k in range(12):
        lt.append(X_final[i][k])
    ans=model.predict(sc.transform(np.array([lt])))
    
    sev.append((ids[i],ans[0]));
        
answers=[]


for i in range(2500):
    answers.append(sev[i][1]);
    
    
    
    


troops=dataset["Troops_Mobilized(thousands)"].values
plt.hist(troops)
plt.show()