# Basketball-Playoff-Prediction

## Aim of this program
The program uses the NBA dataset from Kaggle (https://www.kaggle.com/fivethirtyeight/fivethirtyeight-nba-elo-dataset?select=nbaallelo.csv), asks for a team's first 10 results and predicts whether the team participate in the playoff or not.

## Structure and operation
The program contains three different files (modules).
<br>The first one is responsible for feature engineering and data preparation for machine learning algorithms.
<br>The second file has two classes: train and test. These classes applies different classification algorithms (Decision Trees, Random Forests, Support Vector Classifiers) for training and testing the dataset.
<br>The third file creates a user menu and asks for input - these are the results of the first 10 matches of the season.
<br><br>The program gives back whether the given team would reach the playoff or not.

## Technology and Methodology
<br>For creating and preparing the data: Python Pandas
<br>For training and testing: Python Scikit-Learn
<br>The whole program has been written per OOP standards.
