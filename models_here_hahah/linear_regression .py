
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

class LinearRegressionModel(MetaModel):
    
    def __init__(self):
        super().__init__("Linear Regression")
        
    def preprocess(self, df):
        """
        """
        temp_df = df.copy()

        temp_df['total_rad_1h:J'] = df['diffuse_rad_1h:J'] + df['direct_rad_1h:J']

        temp_df = temp_df.dropna(axis=0, how="all", subset="total_rad_1h:J")

        if('y' in temp_df.columns.tolist()):
            temp_df = temp_df.dropna(axis=0, how="all", subset="y")

        return temp_df

    def train(self, df):
        """
        """
        df = self.preprocess(df)

        self.model = LinearRegression()
        self.model.fit(df['total_rad_1h:J'].values.reshape(-1, 1), df['y'].values.reshape(-1, 1))


    def predict(self, df):
        """
        """
        df = self.preprocess(df)

        y_preds = self.model.predict(df['total_rad_1h:J'].values.reshape(-1, 1))
        y_preds_arr = np.array(y_preds)

        out_df = pd.DataFrame(y_preds_arr, columns=["y_pred"])        
        return out_df
    

df = ml.data.get_training_flattened()
for location in ['A', 'B', 'C']:
    temp_df = df[df['location']==location]

    lr = LinearRegressionModel()
    lr.test(temp_df)

print("Done creating a linear regression model!")