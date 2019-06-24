#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:46:54 2019

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




#scores_train = []
#scores_test = []
# Setting different values for C parametter used later in the LogisticRegression Classifier
C = [0.001, 0.01, 0.1, 1, 5, 20, 50]

#Defining some markers and labels used later on the graph
markers = ['+', 'x', 'o', '*', '^', 'v', '>']
labels = ['C=0.001', 'C=0.01', 'C=0.1', 'C=1', 'C=5', 'C=20', 'C=50']



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
    
    
    
def plot_feature_importances(classifier):
    
    n_features = data_no_label.shape[1]
    plt.barh(np.arange(n_features), classifier.feature_importances_, align='center')
    
    plt.yticks(np.arange(n_features), f_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    
    


#To silence the warnings
import warnings
warnings.filterwarnings("ignore")


# Reading data from the fraud_payment CSV file 
data = pd.read_csv('fraud_payment.csv')


# In the dataset, there is one column that is non-numerical: paymentMethod
# It should be converted to numerical first because many machine learning algorithms require all features to be numeric
# Using get_dummies(), this column (feature) is converted to dummy variables with one-hot encoding 
#paymentMethod is replaced by paymentMethod_creditcard,paymentMethod_paypal,paymentMethod_storecredit
data = pd.get_dummies(data, columns=['paymentMethod'])


# we need to remove the last column which is just a label shows either a payment is fraud or no
data_no_label = data.drop('label', axis=1)



# Reading name of the features (columns) in the dataset
f_names = data_no_label.columns.tolist()


# Spliting the dataset up into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data_no_label, data['label'],
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
plt.ylim(-11, 4)


# Adding labels and legend to the graph
plt.xlabel("Feature Name")
plt.ylabel("Coefficient Magnitude")
plt.legend(loc = 'best') 


# Calling the DecisionTree_Classifirer() function for every value of C
DecisionTree_Classifirer(X_train, y_train, X_test, y_test)






