# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 20:58:31 2020

@author: ucanr
"""
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import pydotplus as pdot
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export
from sklearn.model_selection import train_test_split
import os


# Get the data ready
%matplotlib inline
trainFile = "C:/Users/ucanr/iCloudDrive/Statistics/Statistics for Me/Codes/FuelConsumption.csv"
pwd = os.getcwd()
os.chdir(os.path.dirname(trainFile))
myPredictionData = pd.read_csv(os.path.basename(trainFile))


#Colnames for reference
print(myPredictionData.columns)

#The features we are considering
feature_cols=['FUELCONSUMPTION_HWY','CO2EMISSIONS']

#Cross-validation
train_X, test_X,\
    train_y, test_y=train_test_split(myPredictionData[feature_cols], 
                                     myPredictionData['MAKE'])
    
#Set up the depth list for the tree branches
depth_list=[2,3,4,5,6,7,8]

for depth in depth_list:
    clf_tree=DecisionTreeClassifier(max_depth=depth)
    clf_tree.fit(train_X,train_y)

clf_tree=DecisionTreeClassifier(max_depth=2)
clf_tree.fit(train_X,train_y)

#Apply the test data to the model
tree_predict=clf_tree.predict(test_X)

#Visualize the tree
export_graphviz(clf_tree,out_file='model_tree.odt',
                feature_names=train_X.columns)
model_tree_graph=pdot.graphviz.graph_from_dot_file('model_tree.odt')
model_tree_graph.write_jpg('model_tree.jpg')
from IPython.display import Image
Image(filename='model_tree.jpg')