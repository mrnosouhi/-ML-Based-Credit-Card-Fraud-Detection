#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 5 15:02:10 2019

@author: mohammad
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import itertools





def LogisticRegression_Classifirer(C, x_train, y_train, x_test, y_test, marker, label):
    
    # Initializing and training the LogisticRegression classifiers
    LR = LogisticRegression(C = c)
    LR.fit(x_train, y_train)
    
    #scores_train.append(LR.score(x_train, y_train))
    #scores_test.append(LR.score(x_test, y_test))
    
    print("\nC={}".format(C))
    print("Training set score for C = {}: {:.3f}".format(c, LR.score(x_train, y_train)))
    print("Test set score: {:.3f}".format(LR.score(x_test, y_test)))
    
    
    plt.plot(LR.coef_.T, marker = marker, label=label)
    
#To silence the warnings
import warnings
warnings.filterwarnings("ignore")


# Reading data from the fraud_payment CSV file 
data = pd.read_csv('creditcard.csv')


# In the dataset, there is one column that is non-numerical: paymentMethod
# It should be converted to numerical first because many machine learning algorithms require all features to be numeric
# Using get_dummies(), this column (feature) is converted to dummy variables with one-hot encoding 
#paymentMethod is replaced by paymentMethod_creditcard,paymentMethod_paypal,paymentMethod_storecredit
#data = pd.get_dummies(data, columns=['paymentMethod'])


# we need to remove the last column which is just a label shows either a payment is fraud or no
data_no_label = data.drop('Class', axis=1)



# Reading name of the features (columns) in the dataset
f_names = data_no_label.columns.tolist()


# Spliting the dataset up into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data_no_label, data['Class'],
    test_size=0.33, random_state=0)





plt.subplots(figsize=(6, 6), dpi=100)

# Calling the LogisticRegression_Classifirer() function for every value of C
for i, c in enumerate(C):
    LogisticRegression_Classifirer(c, X_train, y_train, X_test, y_test, markers[i], labels[i])
 
    
# Adding the feature names to xticks    
plt.xticks(range(data_no_label.shape[1]), f_names, rotation=90)


# Drawing a horizontal line for y=0
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])


#Limitting x and y axis
plt.xlim(xlims)
plt.ylim(-2, 2)


# Adding labels and legend to the graph
plt.xlabel("Feature Name")
plt.ylabel("Coefficient Magnitude")
plt.legend(loc = 'best') 
