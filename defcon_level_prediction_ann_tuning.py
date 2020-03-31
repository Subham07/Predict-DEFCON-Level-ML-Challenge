# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:14:20 2020

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


# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 0:10].values #taking indexes from 1st to 9th one barring defcon level and id
y = dataset.iloc[:, 10:11].values # penultimate column storing class

np.unique(y)


# encode class values
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.25, random_state = 0) #test on 2000 observations and train on 8000


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#importing libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #Dropout to reduce overfitting (line 131)

#normal way
classifier= Sequential()

#running the input layer and the first hidden layer with dropout
classifier.add(Dense(output_dim=15, init='uniform', activation='relu', input_dim=10)) #average of no. of nodes in i/p and nodes in o/p

classifier.add(Dense(output_dim=10, init='uniform', activation='relu'))

classifier.add(Dense(output_dim=7, init='uniform', activation='relu')) #average of no. of nodes in i/p and nodes in o/p

classifier.add(Dense(output_dim=9, init='uniform', activation='relu'))


classifier.add(Dense(output_dim=9, init='uniform', activation='relu'))

#adding o/p Layer
classifier.add(Dense(output_dim=5, init='uniform', activation='softmax')) #output node is 1 as binary outcome will exit the bank / wont exit
#sigmoid function for o/p , use softmax if o/p is >1

#compiling the ANN
classifier.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy']) # adam algorithm: adaptive moment estimation, to update weights iteratively
#adam algo comes under stochastic gradient descent
#binary crossentropy as binary outcome

classifier.fit(X_train, y_train, batch_size=25, nb_epoch=250)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

for i in range(2500):
    if(y_pred[i][0]==max(y_pred[i][0],y_pred[i][1],y_pred[i][2],y_pred[i][3],y_pred[i][4])):
        y_pred[i][0]=1;
        y_pred[i][1]=0;
        y_pred[i][2]=0;
        y_pred[i][3]=0;
        y_pred[i][4]=0;
    elif(y_pred[i][1]==max(y_pred[i][0],y_pred[i][1],y_pred[i][2],y_pred[i][3],y_pred[i][4])):
        y_pred[i][1]=1;
        y_pred[i][0]=0;
        y_pred[i][2]=0;
        y_pred[i][3]=0;
        y_pred[i][4]=0;
    elif(y_pred[i][2]==max(y_pred[i][0],y_pred[i][1],y_pred[i][2],y_pred[i][3],y_pred[i][4])):
        y_pred[i][0]=0;
        y_pred[i][1]=0;
        y_pred[i][2]=1;
        y_pred[i][3]=0;
        y_pred[i][4]=0;
    elif(y_pred[i][3]==max(y_pred[i][0],y_pred[i][1],y_pred[i][2],y_pred[i][3],y_pred[i][4])):
        y_pred[i][0]=0;
        y_pred[i][1]=0;
        y_pred[i][2]=0;
        y_pred[i][3]=1;
        y_pred[i][4]=0;
    elif(y_pred[i][4]==max(y_pred[i][0],y_pred[i][1],y_pred[i][2],y_pred[i][3],y_pred[i][4])):
        y_pred[i][0]=0;
        y_pred[i][1]=0;
        y_pred[i][2]=0;
        y_pred[i][3]=0;
        y_pred[i][4]=1;
        


t=0;
f=0;
for i in range(2500):
    if(y_pred[i][0]==y_test[i][0] and y_pred[i][1]==y_test[i][1] and y_pred[i][2]==y_test[i][2] and y_pred[i][3]==y_test[i][3]):
        t=t+1;
    else:
        f=f+1;

print(t/(t+f))

# Importing the test dataset
dataset_test = pd.read_csv('test.csv')

X_final=dataset_test.iloc[:, 0:10].values;
ids=dataset_test.iloc[:, 10:11].values;





sev=[];
for i in range(2500):
    lt=[]
    for k in range(10):
        lt.append(X_final[i][k])
    ans=classifier.predict(sc.transform(np.array([lt])));
    if(ans[0][0]==max(ans[0][1],ans[0][0],ans[0][2],ans[0][3],ans[0][4])):
        ans[0][0]=1;
        ans[0][1]=0;
        ans[0][2]=0;
        ans[0][3]=0;
        ans[0][4]=0;
    elif(ans[0][1]==max(ans[0][1],ans[0][0],ans[0][2],ans[0][3],ans[0][4])):
        ans[0][0]=0;
        ans[0][1]=1;
        ans[0][2]=0;
        ans[0][3]=0;
        ans[0][4]=0;
    elif(ans[0][2]==max(ans[0][1],ans[0][0],ans[0][2],ans[0][3],ans[0][4])):
        ans[0][0]=0;
        ans[0][1]=0;
        ans[0][2]=1;
        ans[0][3]=0;
        ans[0][4]=0;
    elif(ans[0][3]==max(ans[0][1],ans[0][0],ans[0][2],ans[0][3],ans[0][4])):
        ans[0][0]=0;
        ans[0][1]=0;
        ans[0][2]=0;
        ans[0][3]=1;
        ans[0][4]=0;
    elif(ans[0][4]==max(ans[0][1],ans[0][0],ans[0][2],ans[0][3],ans[0][4])):
        ans[0][0]=0;
        ans[0][1]=0;
        ans[0][2]=0;
        ans[0][3]=0;
        ans[0][4]=1;
        
    if(ans[0][0]==1):
        sev.append((ids[i],0));
    elif(ans[0][1]==1):
        sev.append((ids[i],1));
    elif(ans[0][2]==1):
        sev.append((ids[i],2));
    elif(ans[0][3]==1):
        sev.append((ids[i],3));
    elif(ans[0][4]==1):
        sev.append((ids[i],4));
        
answers=[]


for i in range(2500):
    if(sev[i][1]==0):
        answers.append(1)
    elif(sev[i][1]==1):
        answers.append(2);
    elif(sev[i][1]==2):
        answers.append(3)
    elif(sev[i][1]==3):
        answers.append(4);
    elif(sev[i][1]==4):
        answers.append(5);