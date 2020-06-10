# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 08:47:14 2020

@author: patak
"""

import pandas as pd
import numpy as np
from data import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

class Train(Data):
    
    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    
    def logclf(self):
        self.log_clf = LogisticRegression(solver="lbfgs", random_state=42)
        self.score1 = cross_val_score(self.log_clf, self.X_train, self.y_train, verbose=3)
        return self.score1.mean()
    
    def svc(self):
        self.svc_clf = SVC(gamma='auto')
        self.score2 = cross_val_score(self.svc_clf, self.X_train, self.y_train, verbose=3)
        return self.score2.mean()
    
    def linear_svc(self):
        self.linear_svc = LinearSVC()
        self.score3 = cross_val_score(self.linear_svc, self.X_train, self.y_train, verbose=3)
        return self.score3.mean()
    
    def treeclf(self):
        self.tree_clf = DecisionTreeClassifier(max_depth=5)
        self.score4 = cross_val_score(self.tree_clf, self.X_train, self.y_train, verbose=3)
        return self.score4.mean()
    
    def random(self):
        self.rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
        self.score5 = cross_val_score(self.rnd_clf, self.X_train, self.y_train)
        return self.score5.mean()
    
class Test(Data):
    
    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    
    def logclf(self):
        self.log_clf = LogisticRegression(solver="lbfgs", random_state=42)
        self.log_clf.fit(self.X_train, self.y_train.values.ravel())
        self.y_pred = self.log_clf.predict(self.X_test)
        print(confusion_matrix(self.y_test.values.ravel(),self.y_pred))
        print("Precision: {:.2f}%".format(100 * precision_score(self.y_test.values.ravel(),self.y_pred)))
        print("Recall: {:.2f}%".format(100 * recall_score(self.y_test.values.ravel(),self.y_pred)))
        print("F1: {:.2f}%".format(100 * f1_score(self.y_test.values.ravel(),self.y_pred)))
    
    def svc(self):
        self.svc_clf = SVC(gamma='auto')
        self.svc_clf.fit(self.X_train, self.y_train)
        self.y_pred = self.svc_clf.predict(self.X_test)
        print(confusion_matrix(self.y_test,self.y_pred))
        print("Precision: {:.2f}%".format(100 * precision_score(self.y_test,self.y_pred)))
        print("Recall: {:.2f}%".format(100 * recall_score(self.y_test,self.y_pred)))
        print("F1: {:.2f}%".format(100 * f1_score(self.y_test,self.y_pred)))
        
    def linear_svc(self):
        self.linear_svc = LinearSVC()
        self.linear_svc.fit(self.X_train, self.y_train)
        self.y_pred = self.linear_svc.predict(self.X_test)
        print(confusion_matrix(self.y_test,self.y_pred))
        print("Precision: {:.2f}%".format(100 * precision_score(self.y_test,self.y_pred)))
        print("Recall: {:.2f}%".format(100 * recall_score(self.y_test,self.y_pred)))
        print("F1: {:.2f}%".format(100 * f1_score(self.y_test,self.y_pred)))
    
    def treeclf(self):
        self.tree_clf = DecisionTreeClassifier(max_depth=5)
        self.tree_clf.fit(self.X_train, self.y_train)
        self.y_pred = self.tree_clf.predict(self.X_test)
        print(confusion_matrix(self.y_test,self.y_pred))
        print("Precision: {:.2f}%".format(100 * precision_score(self.y_test,self.y_pred)))
        print("Recall: {:.2f}%".format(100 * recall_score(self.y_test,self.y_pred)))
        print("F1: {:.2f}%".format(100 * f1_score(self.y_test,self.y_pred)))
    
    def random(self):
        self.rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
        self.rnd_clf.fit(self.X_train, self.y_train)
        self.y_pred = self.rnd_clf.predict(self.X_test)
        print(confusion_matrix(self.y_test,self.y_pred))
        print("Precision: {:.2f}%".format(100 * precision_score(self.y_test,self.y_pred)))
        print("Recall: {:.2f}%".format(100 * recall_score(self.y_test,self.y_pred)))
        print("F1: {:.2f}%".format(100 * f1_score(self.y_test,self.y_pred)))
    
    
    
    
    
    
    