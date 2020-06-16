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


class Menu:
    """Display menu and respond to choices"""
    
    def __init__(self):
        self.data = Data()
        self.train = Train()
        self.Test = Test()
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
        self.lst = list()
        self.team = input("Enter the name of the team: ")
        self.lst1 = input("Enter the result of the 1st match (W/L): ")
        self.lst.append(self.lst1)
        self.lst2 = input("Enter the result of the 1nd match (W/L): ")
        self.lst.append(self.lst2)
        self.lst3 = input("Enter the result of the 3rd match (W/L): ")
        self.lst.append(self.lst3)
        self.lst4 = input("Enter the result of the 4th match (W/L): ")
        self.lst.append(self.lst4)
        self.lst5 = input("Enter the result of the 5th match (W/L): ")
        self.lst.append(self.lst5)
        self.lst6 = input("Enter the result of the 6th match (W/L): ")
        self.lst.append(self.lst6)
        self.lst7 = input("Enter the result of the 7th match (W/L): ")
        self.lst.append(self.lst7)
        self.lst8 = input("Enter the result of the 8th match (W/L): ")
        self.lst.append(self.lst8)
        self.lst9 = input("Enter the result of the 9th match (W/L): ")
        self.lst.append(self.lst9)
        self.lst10 = input("Enter the result of the 10th match (W/L): ")
        self.lst.append(self.lst10)
        
        self.X_values = pd.DataFrame(columns = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10'])
        self.X_values.loc[0]=self.lst
        
        print(self.X_values)
        
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # self.log_clf = LogisticRegression(solver="lbfgs", random_state=42)
        # self.log_clf.fit(self.X_train, self.y_train.values.ravel())
        # self.y_predict = self.log_clf.predict(self.X_values)
        # return self.y_predict
        
    def quit(self):
        print("Thank you for using NBA Prediction!")
        sys.exit(0)

if __name__=="__main__":
    Menu().run()
        
        