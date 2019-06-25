#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:42:48 2019

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



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def DecisionTree_Classifirer(x_train, y_train, x_test, y_test):
    tree = DecisionTreeClassifier(random_state=0, )
    tree.fit(X_train, y_train)
    
    print("\n***Decision Tree Classifier***")
    print("\nDecision tree Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
    print("Decision tree Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
    
    print("Decision tree depth:")
    print(tree.tree_.max_depth)
    
    print("Decision tree Feature importances:")
    print(tree.feature_importances_)
    plt.subplots(figsize=(5, 5), dpi=100)
    plot_feature_importances(tree)
    
    ## Make predictions on test set
    y_pred = tree.predict(X_test)

    ## Comparing the predictions on the test set with real labels
    print("Accuracy on the test set: {:.3f}".format(accuracy_score(y_pred, y_test)))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    #Plotting confusion matrix based on the real labels and the predictions on the test set 
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cm, class_names,
                          normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
    plt.show()




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



# Calling the DecisionTree_Classifirer() function for every value of C
DecisionTree_Classifirer(X_train, y_train, X_test, y_test)    
    
