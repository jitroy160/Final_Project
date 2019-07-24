# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Abhijit)s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

testingDataset = pd.read_csv('C:\\Users\\hp\\Desktop\\Henry Harvin\\Assignment #6\\test.csv')
trainingDataset = pd.read_csv('C:\\Users\\hp\\Desktop\\Henry Harvin\\Assignment #6\\train.csv')
print (trainingDataset.head())
df = pd.concat([trainingDataset, testingDataset],ignore_index=True, sort=False)
print (df.shape)
print (df.columns)
y = trainingDataset['Loan_Status']

print ("TRAINING DATA DETAILS")
print ("Total  number of records present in the dataset -", trainingDataset.shape[0])
print ("Total  number of columns present in the dataset -", trainingDataset.shape[1])

print ("\n TESTING DATA DETAILS")
print ("Total  number of records present in the dataset -", testingDataset.shape[0])
print ("Total  number of columns present in the dataset -", testingDataset.shape[1])

print ("Following are the columns present in the dataset - ", trainingDataset.columns)
print (trainingDataset.dtypes)

print ("TOTAL NUMBER OF RECORDS IN THE COMBINED DATASET  - ", combined.shape[0])
print ("\n")
categoricalColNames = df.iloc[:,1:].select_dtypes(include=['object'])
requiredCategoricalVariables = list(categoricalColNames.columns.values)
for x in requiredCategoricalVariables:
    print ("Number of value counts for -", x)
    print (df[x].value_counts())
    print ('Number of Missing values: %d'% sum(df[x].isnull()))
    print ("\n")
    
numericalColNames = df.iloc[:,1:].select_dtypes(include=['int64','float64'])
requiredCategoricalVariables = list(numericalColNames.columns.values)
for x in requiredCategoricalVariables:
    print ('Number of missing values in ', x ,': %d'% sum(df[x].isnull()))
    
df.info()    

requiredColumns = list(df.columns.values)
print ("Checking if there are any missing values in the dataset - ")
for col in requiredColumns:
    print ("column name  -", col)
    print ('Final #missing: %d'% sum(df[col].isnull()))
    print ("\n")
    
plt.figure(figsize=(12,12))
sns.heatmap(df.iloc[:, 2:].corr(), annot=True, square=True, cmap='BuPu')
plt.show()

plt.figure(figsize=(20,20))
temp = trainingDataset.iloc[:,2:].select_dtypes(include=['int64','float64'])
requiredColumns = list(temp.columns.values)
counter = 1
for col in requiredColumns:
    plt.subplot(3, 3, counter)
    trainingDataset[col].hist(color = 'green')
    plt.title(col)
    counter = counter + 1
    
plt.figure(figsize=(10,10))
temp2 = pd.crosstab(trainingDataset['Credit_History'], trainingDataset['Loan_Status'])
temp2.plot(kind='bar', stacked=True, color=['orange','blue'], grid=False)

nrow_test = testingDataset.shape[0]
X_test = df[nrow_train:]
X_train = df[:nrow_train]

var_mod = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status','Dependents']
le = LabelEncoder()
for i in var_mod:
    X_test[i] = le.fit_transform(X_test[i])
    X_train[i] = le.fit_transform(X_train[i])
print ("CONVERTED THE CATEGORICAL VARIABLES INTO NUMERICALS")

sns.pairplot(trainingDataset[trainingDataset.columns.values], hue='Loan_Status', diag_kind='kde', height=2);

### THE BOX PLOT SHOW THE OUTLIERS IN YOUR DATA. 
### AS YOU CAN SEE COLUMNS NAMED "APPLICANT INCOME" AND "CO APPLICANT INCOME" HAVE OUTLIERS
temp = trainingDataset.iloc[:,2:].select_dtypes(include=['int64','float64'])
requiredColumns = list(temp.columns.values)
plt.figure(figsize=(10,10))
#trainingDataset[trainingDataset.columns.values].plot.box();
sns.boxplot(data=X_train[requiredColumns], palette="Set2")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split #For K-fold cross validation

from sklearn import metrics

X = X_train.iloc[:, 2:11].values
y = X_train.iloc[:, 12].values
#X = X.reshape(X.shape[0],1)
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.20, random_state=0)

LR_model = LogisticRegression(solver='sag')
LR_model.fit(X_tr,y_tr)
#Make predictions on training set:
predictions = LR_model.predict(X_te)
