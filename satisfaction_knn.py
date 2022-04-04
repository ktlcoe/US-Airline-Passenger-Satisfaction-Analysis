# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 14:55:22 2021

@author: ktlco
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:29:05 2021

@author: ktlco
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def tune_model(n):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train, y_train)
    train_acc = knn.score(X_train, y_train)
    print(str(n) + "\t" + str(train_acc))    
    return 1

path = './data/satisfaction.csv'
df = pd.read_csv(path)

dat = df.drop(["id", "satisfaction_v2"], axis=1)
y = df[["satisfaction_v2"]]

# convert customer type, gender, and type of travel into dummy variables
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(dat[["Gender"]]).toarray())
dat = dat.join(enc_df)
dat = dat.rename(columns={0:"Female", 1:"Male"})
enc_df = pd.DataFrame(enc.fit_transform(dat[["Customer Type"]]).toarray())
dat = dat.join(enc_df)
dat = dat.rename(columns={0:"Loyal Customer", 1:"Disloyal Customer"})
enc_df = pd.DataFrame(enc.fit_transform(dat[["Type of Travel"]]).toarray())
dat = dat.join(enc_df)
dat = dat.rename(columns={0:"Personal Travel", 1:"Business Travel"})
enc_df = pd.DataFrame(enc.fit_transform(dat[["Class"]]).toarray())
dat = dat.join(enc_df)
dat = dat.rename(columns={0:"Eco", 1:"Business", 2:"Eco Plus"})
dat = dat.drop(["Gender", "Customer Type", "Type of Travel", "Class"], axis=1)

# convert satisfaction into 1's for satisfied, 0's for not satisfied
y.loc[y["satisfaction_v2"]=="neutral or dissatisfied"] = 0
y.loc[y["satisfaction_v2"]=="satisfied"] = 1
y = y.loc[:,"satisfaction_v2"].values.astype('int')

# pre-processing - fill missing values in arrival delay with 0's and scale data
dat[dat["Arrival Delay in Minutes"].isnull()] = 0
ndata = preprocessing.scale(dat)

# perform train/test split before assessing models
X_train, X_test, y_train, y_test = train_test_split(ndata, y, test_size=.2, random_state=5)

# train model
print("k \t test acc")
print("=============")
for n in range(2,6):
    tune_model(n)

# test model
n = 3
knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_train, y_train)
test_acc = knn.score(X_test, y_test)
print("KNN test accuracy for n = " + str(n) + ": " + str(test_acc))










