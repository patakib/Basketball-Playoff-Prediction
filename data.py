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
        
        # self.trh = self.df[(self.df['TeamYear']=='1947-TRH') & (self.df['is_playoffs']==0)]
        # self.trh = self.trh[['game_result']]
        # self.trh = self.trh.head(20)
        # self.trh = self.trh.rename(columns={'game_result':'1947-TRH'})
        # self.trh = self.trh.T
        
        for team in self.allteams:
            self.teamrecord = self.df[(self.df['TeamYear']==team) & (self.df['is_playoffs']==0)]
            self.teamrecord = self.teamrecord[['game_result']]
            self.teamrecord = self.teamrecord.head(20)
            self.teamrecord = self.teamrecord.rename(columns={'game_result': team})
            self.teamrecord = self.teamrecord.T
            self.database = pd.concat([self.teamlabels, self.teamrecord], axis=1, sort=False)
            lst = [self.teamrecord]
            del lst
            
        

    def overview(self, rows):
        return self.df.head(rows)
    
    def playoffdata(self,rows):
        return self.teamlabels.head(rows)
    
    
        