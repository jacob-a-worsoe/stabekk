
import warnings

# Temporarily suppress FutureWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# In module1.py
import sys
import os

# Get the absolute path of folder2
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, os.pardir)
folder2_dir = os.path.join(parent_dir, 'ml_combat')

# Add folder2 to sys.path
sys.path.append(parent_dir)

from ml_combat.MetaModel import MetaModel
import ml_combat as ml

###### Start Here ######
class NaiveModel(MetaModel):
    
    def __init__(self):
        super().__init__("Naive Model")
        
    def preprocess(self, df):
        """
        """
        temp_df = df.copy()

        return temp_df

    def train(self, df):
        """
        """
        df = self.preprocess(df)

        self.model = df.y.mean()


    def predict(self, df):
        df = self.preprocess(df)

        df['y_pred'] = self.model

        return df[['ds', 'y_pred']].copy()
    


df = ml.data.get_training_flattened()

for location in ['A', 'B', 'C']:
    temp_df = df[df['location'] == location]

    lr = NaiveModel()
    lr.test(temp_df)

print("Done creating a linear regression model!")