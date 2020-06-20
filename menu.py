# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 08:14:11 2020

@author: patak
"""

import sys
from data import Data
from machine_learning import Train, Test
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

x = 1

class Menu:
    """Display menu and respond to choices"""

    
    def __init__(self):
        self.data = Data()
        self.train = Train()
        self.test = Test()
        self.choices = {
            "1": self.prediction,
            "2": self.quit}
    
    def display_menu(self):
        print(
            """
            NBA Prediction Menu
            
            1. Predict Playoff
            2. Quit
            """
            )
        
    def run(self):
        
        while True:
            self.display_menu()
            choice = input("Enter an option: ")
            action = self.choices.get(choice)
            if action:
                action()
            else:
                print("{0} is not a valid choice".format(choice))
                
    def prediction(self):
        
    
        
        self.X_valueslist=['M1_W', 'M1_L', 'M2_W', 'M2_L', 'M3_W', 'M3_L', 'M4_W', 'M4_L', 'M5_W', 'M5_L', 'M6_W', 'M6_L', 'M7_W', 'M7_L', 'M8_W', 'M8_L', 'M9_W', 'M9_L', 'M10_W', 'M10_L']
        self.lst = [0,1,2,3,4,5,6,7,8,9]
        self.results=[]
        
        self.team = input("Enter the name of the team: ")
        
        for value in self.lst:
            a = input("Enter the result (W/L): ")
            if a == 'W':
                self.results.append([1,0])
            elif a == 'L':
                self.results.append([0,1])
                
        self.resultlist = [item for elem in self.results for item in elem]
        self.result_dict = dict(zip(self.X_valueslist, self.resultlist))

        self.X_values = pd.DataFrame([self.result_dict])
        
        self.test.X_train, self.test.X_test, self.test.y_train, self.test.y_test = train_test_split(self.test.X, self.test.y, test_size=0.2, random_state=42)
        self.log_clf = LogisticRegression(solver="lbfgs", random_state=42)
        self.log_clf.fit(self.test.X_train, self.test.y_train.values.ravel())
        self.y_predict = self.log_clf.predict(self.X_values)
        print("Our prediction: ")
        print(self.y_predict)
        
        
        
    def quit(self):
        print("Thank you for using NBA Prediction!")
        sys.exit(0)

if __name__=="__main__":
    Menu().run()
        
        