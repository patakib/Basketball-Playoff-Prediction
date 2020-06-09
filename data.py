# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:40:45 2020

@author: patak
"""

import pandas as pd
import numpy as np

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
        
        self.df = pd.read_csv('nbaallelo.csv')
        self.df['TeamYear'] = self.df['year_id'].astype(str) + '-' + self.df['team_id']
        self.df_playoff = self.df[self.df['is_playoffs']==1]
        
        self.playoffteams = self.df_playoff['TeamYear'].unique().tolist()
        self.allteams = self.df['TeamYear'].unique().tolist()
        self.labels = list()
        
        for team in self.allteams:
            if team in self.playoffteams:
                self.labels.append(1)
            else:
                self.labels.append(0)
        
        self.teamlabels = pd.DataFrame(
            {'TeamYear': self.allteams,
             'Playoff': self.labels
             })
        
        self.datacolumns = ["M1","M2","M3","M4","M5","M6","M7","M8","M9","M10"]
        # ,"M11","M12","M13","M14","M15","M16","M17","M18","M19","M20"
        for column in self.datacolumns:
            self.teamlabels[column] = np.nan
        
        
        # self.teamlabels = self.teamlabels.set_index('TeamYear')
        # self.trh = self.df[(self.df['TeamYear']=='1947-TRH') & (self.df['is_playoffs']==0)]
        # self.trh = self.trh[['game_result']]
        # self.trh = self.trh.head(20)
        # self.trh = self.trh.rename(columns={'game_result':'1947-TRH'})
        # self.trh = self.trh.T
        # self.trh.columns = self.datacolumns
        # self.teamlabels.update(self.trh)
        
        self.teamlabels = self.teamlabels.set_index('TeamYear')
        
        for team in self.allteams:
            self.teamrecord = self.df[(self.df['TeamYear']==team) & (self.df['is_playoffs']==0)]
            self.teamrecord = self.teamrecord[['game_result']]
            self.teamrecord = self.teamrecord.head(10)
            self.teamrecord = self.teamrecord.rename(columns={'game_result': team})
            self.teamrecord = self.teamrecord.T
            self.teamrecord.columns = self.datacolumns
            self.teamlabels.update(self.teamrecord)
            lst = [self.teamrecord]
            del lst
            
        

    def overview(self, rows):
        return self.df.head(rows)
    
    def playoffdata(self,rows):
        return self.teamlabels.head(rows)
    
    
        