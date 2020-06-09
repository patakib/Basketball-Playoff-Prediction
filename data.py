# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:40:45 2020

@author: patak
"""

import pandas as pd

class RawData:
    """loading the dataset into pandas dataframe"""
    
    def __init__(self):
        """Initialize the dataframe"""
        self.df = pd.read_csv('nbaallelo.csv')
        
    def overview(self, rows):
        return self.df.head(rows)
    

class Data:
    """Form and reshape the rawdata to fit our purposes"""
    
    def __init__(self):
        """Initialize the dataframe"""
        self.df = pd.read_csv('nbaallelo.csv')
        self.df['TeamYear'] = self.df['year_id'].astype(str) + '-' + self.df['team_id']

    def overview(self, rows):
        return self.df.head(rows)
        
        