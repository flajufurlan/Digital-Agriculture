#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Fist step is to install the mlxtend library to perform step forward feature selection 
# Random Forest classifier for feature selection and model building


# In[2]:


conda install -c conda-forge mlxtend 


# In[3]:


# The first steps are too make imports, load the dataset, and split it into training and testing sets.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import os


# In[4]:


# Reading the data (csv format)
df = pd.read_csv('winequality-white.csv', sep=';')


# In[5]:


#Summary statistics

FF_Description = df.describe()
FF_Description.to_csv("FF_Description.csv")


# In[6]:


# In this command a train/test aplit.
#Train Dataset: Used to fit the machine learning model.
#Test Dataset: Used to evaluate the fit machine learning model.
X_train, X_test, y_train, y_test = train_test_split(
    df.values[:,:-1],
    df.values[:,-1:],
    test_size=0.25,
    random_state=42)

y_train = y_train.ravel()
y_test = y_test.ravel()

# Showing the traning/test data set shape
print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)


# In[7]:


#Defining a classifier 
#Using random forest that is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging
#n_estimatorsint, default=100. The number of trees in the forest.
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)


# In[8]:


# Build step forward feature selection giving the subset of features that in this case are k_features=5
# floating algorithms have an additional exclusion or inclusion step to remove features once they were included (or excluded)
# The verbose is defined for mlxtend to report
# The scoring to accuracy is used to score the models results that were built based on the selected features 
# mlxtend feature selector uses cross validation internally, and we set our desired folds to 5.
sfs1 = sfs(clf,
           k_features=5,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)


# In[9]:


# Perform SFFS
# The score metric got comes from the subset of 5 features , using cross validation
sfs1 = sfs1.fit(X_train, y_train)


# In[10]:


# The commands are showing which features were selected for the model
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)


# In[11]:


# With the selected features it is possible not to build a full with the traning and test sets 
# The command is building a classifier for only the subset pf the selected features

clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
clf.fit(X_train[:, feat_cols], y_train)

y_train_pred = clf.predict(X_train[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))


# In[12]:


# Comparision of the model above with the accuracies of another full model using all features 
# Comparision check 
clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test)
print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))


# In[13]:


# It is important to check the feature subset that will work best for the data and do the comparision with the full data set.
# Comparing the two models the accuracy is not very high and is very similar


# In[16]:


# With six features
# Build step forward feature selection
# It is important to choose the right numbers of features and check because 
#it can lead to a sub-optimak numer and combination of features being decided upon 
sfs1 = sfs(clf,
           k_features=6,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)


# In[ ]:


# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)


# In[19]:


# Which features?
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)


# In[20]:


# Build full model with selected features
clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
clf.fit(X_train[:, feat_cols], y_train)

y_train_pred = clf.predict(X_train[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))


# In[21]:


# the score with 6 (0.64) features a little bit higher than with 5 (0.62) features. 
# The full model built with the selected features ( training an testing accuracy) 5 and 6 are very approximate 

