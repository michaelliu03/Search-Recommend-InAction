#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:train.py
# @Author: Michael.liu
# @Date:2020/6/3 17:42
# @Desc: this code is avaza_ctr

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn import  metrics
from .utils import load_df

def print_metrics(true_values, predicted_values):
    print("Accuracy",metrics.accuracy_score(true_values,predicted_values))
    print("AUC",metrics.roc_auc_score(true_values,predicted_values))
    print("Confusion Matrix",metrics.confusion_matrix(true_values,predicted_values))
    print(metrics.classification_report(true_values,predicted_values))

def classify(classifier_class,train_input,train_targets):
    classifier_object = classifier_class()
    classifier_object.fit(train_input,train_targets)
    return classifier_object

def save_model(clf):
    joblib.dump(clf,'lr_classifier.pkl')

train_data = load_df('avazu_ctr/train_sample.csv')
X_train,X_test,y_train,y_test = train_test_split(train_data[0::,1::],train_data[0::,0],test_size=0.3,random_state=0)

classifier =  classify(LogisticRegression,X_train,y_train)
predictions = classifier.predict(X_test)
print_metrics(y_test,predictions)
save_model(classifier)