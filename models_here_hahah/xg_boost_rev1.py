# Import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
from itertools import combinations

# Update sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, os.pardir)
sys.path.append(parent_dir)

# Import modules
from ml_combat.MetaModel import MetaModel
import ml_combat as ml

class XGBoostModel(MetaModel):

    def __init__(self):
        super().__init__("XGBoost")

    def preprocess(self, df):
        temp_df = df.copy()
        temp_df['location'] = pd.factorize(temp_df['location'])[0]

        if 'diffuse_rad_1h:J' in df.columns and 'direct_rad_1h:J' in df.columns:
            temp_df['total_rad_1h:J'] = df['diffuse_rad_1h:J'] + df['direct_rad_1h:J']

        temp_df.dropna(axis=0, how="all", subset=["total_rad_1h:J"], inplace=True)

        if 'ds' in temp_df.columns:
            try:
                temp_df['ds'] = pd.to_datetime(temp_df['ds'])
                temp_df['hour_of_day'] = temp_df['ds'].dt.hour
                temp_df['month'] = temp_df['ds'].dt.month
            except:
                print("Error converting ds to datetime")

        if 'y' in temp_df.columns:
            temp_df.dropna(axis=0, how="all", subset=["y"], inplace=True)

        selected_cols = [
            'location', 
            'total_rad_1h:J',
            'hour_of_day', 
            'month', 
            'clear_sky_energy_1h:J', 
            'sun_elevation:d'
        ]

        if 'y' in df.columns:
            selected_cols.append('y')

        return temp_df[selected_cols]

    def train(self, df):
        df = self.preprocess(df)
        features = [col for col in df.columns if col != 'y']
        X = df[features]
        y = df['y']
        self.model = xgb.XGBRegressor()
        self.model.fit(X, y)

    def predict(self, df):
        df = self.preprocess(df)
        features = [col for col in df.columns if col != 'y']
        X = df[features]
        y_preds = self.model.predict(X)
        return pd.DataFrame({'y_pred': y_preds})


df = ml.data.get_training_flattened()
test_df = ml.data.get_testing_flattened()
print(df.columns)

ret = pd.DataFrame()

for location in ['A', 'B', 'C']:
    temp_df = df[df['location']==location]
    xgboost_model = XGBoostModel()
    xgboost_model.test(temp_df)
    xgboost_model.train(temp_df) 

    forecast = xgboost_model.predict(test_df[test_df['location'] == location])
    ret = pd.concat([ret, forecast])

print("Done creating an XGBoost model!")

ret = ret.reset_index(drop=True).reset_index().rename(columns={'index': 'id', 'y_pred': 'prediction'})
ret.prediction = ret.prediction.apply(lambda a : max(a, 0))
ret.to_csv("xg_boost_rev1.csv", index=False)


"""

all_columns = df.columns.tolist()
# Assuming 'y' is your target column, you'd exclude it from the predictors list.
all_columns.remove('y')
five_column_combinations = list(combinations(all_columns, 5))
best_score = float('inf')  # or float('-inf') if you're looking for the maximum score
best_combination = None

for combo in five_column_combinations:
    # Subset your dataframe
    selected_columns = list(combo) + ['y'] +['ds']
    temp_df = df[selected_columns]

    total_score = 0
    for location in ['A', 'B', 'C']:
        temp_df = df[df['location']==location]
        xgboost_model = XGBoostModel()
        xgboost_model.train(temp_df)
        #print("testing on df with these columns", temp_df.columns)  
        total_score += sum(xgboost_model.test(temp_df))
    
    # Check if this combination has better performance
    if total_score < best_score:  # or score > best_score if looking for maximum score
        best_score = total_score
        best_combination = combo

print(f"The best columns are: {best_combination} with a score of: {best_score}")


"""