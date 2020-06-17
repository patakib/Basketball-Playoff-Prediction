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
        
        self.X_values = pd.DataFrame(columns = ['M1_W', 'M1_L', 'M2_W', 'M2_L', 'M3_W', 'M3_L', 'M4_W', 'M4_L', 'M5_W', 'M5_L', 'M6_W', 'M6_L', 'M7_W', 'M7_L', 'M8_W', 'M8_L', 'M9_W', 'M9_L', 'M10_W', 'M10_L', ])
        
        self.team = input("Enter the name of the team: ")
        self.X_values1 = input("Enter the result of the 1st match (W/L): ")
        if self.X_values1 == 'W':
            self.X_values.at[0,'M1_W']=1
            self.X_values.at[0,'M1_L']=0
        elif self.X_values1 == 'L':
            self.X_values.at[0,'M1_L']=1
            self.X_values.at[0,'M1_W']=0
            
        self.X_values2 = input("Enter the result of the 2nd match (W/L): ")
        if self.X_values2 == 'W':
            self.X_values.at[0,'M2_W']=1
            self.X_values.at[0,'M2_L']=0
        elif self.X_values2 == 'L':
            self.X_values.at[0,'M2_L']=1
            self.X_values.at[0,'M2_W']=0
            
        self.X_values3 = input("Enter the result of the 3rd match (W/L): ")
        if self.X_values3 == 'W':
            self.X_values.at[0,'M3_W']=1
            self.X_values.at[0,'M3_L']=0
        elif self.X_values3 == 'L':
            self.X_values.at[0,'M3_L']=1
            self.X_values.at[0,'M3_W']=0
            
        self.X_values4 = input("Enter the result of the 4th match (W/L): ")
        if self.X_values4 == 'W':
            self.X_values.at[0,'M4_W']=1
            self.X_values.at[0,'M4_L']=0
        elif self.X_values4 == 'L':
            self.X_values.at[0,'M4_L']=1
            self.X_values.at[0,'M4_W']=0
            
        self.X_values5 = input("Enter the result of the 5th match (W/L): ")
        if self.X_values5 == 'W':
            self.X_values.at[0,'M5_W']=1
            self.X_values.at[0,'M5_L']=0
        elif self.X_values5 == 'L':
            self.X_values.at[0,'M5_L']=1
            self.X_values.at[0,'M5_W']=0
            
        self.X_values6 = input("Enter the result of the 6th match (W/L): ")
        if self.X_values6 == 'W':
            self.X_values.at[0,'M6_W']=1
            self.X_values.at[0,'M6_L']=0
        elif self.X_values6 == 'L':
            self.X_values.at[0,'M6_L']=1
            self.X_values.at[0,'M6_W']=0
            
        self.X_values7 = input("Enter the result of the 7th match (W/L): ")
        if self.X_values7 == 'W':
            self.X_values.at[0,'M7_W']=1
            self.X_values.at[0,'M7_L']=0
        elif self.X_values7 == 'L':
            self.X_values.at[0,'M7_L']=1
            self.X_values.at[0,'M7_W']=0
            
        self.X_values8 = input("Enter the result of the 8th match (W/L): ")
        if self.X_values8 == 'W':
            self.X_values.at[0,'M8_W']=1
            self.X_values.at[0,'M8_L']=0
        elif self.X_values8 == 'L':
            self.X_values.at[0,'M8_L']=1
            self.X_values.at[0,'M8_W']=0
            
        self.X_values9 = input("Enter the result of the 9th match (W/L): ")
        if self.X_values9 == 'W':
            self.X_values.at[0,'M9_W']=1
            self.X_values.at[0,'M9_L']=0
        elif self.X_values9 == 'L':
            self.X_values.at[0,'M9_L']=1
            self.X_values.at[0,'M9_W']=0
            
        self.X_values10 = input("Enter the result of the 10th match (W/L): ")
        if self.X_values10 == 'W':
            self.X_values.at[0,'M10_W']=1
            self.X_values.at[0,'M10_L']=0
        elif self.X_values10 == 'L':
            self.X_values.at[0,'M10_L']=1
            self.X_values.at[0,'M10_W']=0
        
        # print(self.X_values)
        
        self.test.X_train, self.test.X_test, self.test.y_train, self.test.y_test = train_test_split(self.test.X, self.test.y, test_size=0.2, random_state=42)
        self.log_clf = LogisticRegression(solver="lbfgs", random_state=42)
        self.log_clf.fit(self.test.X_train, self.test.y_train.values.ravel())
        self.y_predict = self.log_clf.predict(self.X_values)
        print(self.y_predict)
        
    def quit(self):
        print("Thank you for using NBA Prediction!")
        sys.exit(0)

if __name__=="__main__":
    Menu().run()
        
        