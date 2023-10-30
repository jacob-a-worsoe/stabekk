
# Standard imports and settings
if True:
    import warnings

    # Temporarily suppress FutureWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

    import pandas as pd
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

import xgboost as xgb
from sklearn.model_selection import train_test_split


class XGBoostHenrik(MetaModel):
    
    def __init__(self):
        super().__init__("XGBoost Henrik")
        self.features = []
        
    def preprocess(self, df: pd.DataFrame):
        """
        """
        temp_df = df.copy()

        has_target = 'y' in temp_df.columns

        self.features.extend(['month',
                             'hour',
                            'total_rad_1h:J',
                            'fresh_snow_12h:cm',
                            'snow_water:kgm2',
                            'is_day:idx',
                            'is_in_shadow:idx',
                            'rain_water:kgm2',
                            'sun_azimuth:d',
                            'sun_elevation:d',
                            't_1000hPa:K',
                            'dew_or_rime:idx',
                            'air_density_2m:kgm3',
                            'absolute_humidity_2m:gm3'])
        
        # TEMP
        #self.features = ['total_rad_1h:J', 'y']
            
        # FEATURE ENGINEERING

        temp_df['total_rad_1h:J'] = temp_df['diffuse_rad_1h:J'] + temp_df['direct_rad_1h:J']    
        temp_df['hour'] = temp_df['ds'].dt.hour
        temp_df['month'] = temp_df['ds'].dt.month

        
        # SETTING NAN TO 0 CONFORMING TO XGBOOST
        temp_df.fillna(0, inplace=True)

        # DROPPING UNEEEDED FEATURES
        if(has_target):
            features_w_y = self.features + ['y']
            print("FEATUERES WITH Y", features_w_y)
            temp_df = temp_df[features_w_y].copy()

        else:
            temp_df = temp_df[self.features].copy()

        return temp_df

    def train(self, df):
        """
        """

        temp_df = self.preprocess(df)

        # Separate features and target
        features = temp_df.drop('y', axis=1, inplace=False).copy()
        target = temp_df['y'].copy()

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.7)

        # Setup XGB
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='mae'
        )

        self.model.fit(
            X_train,
            y_train,
            verbose=True,
            eval_set=[(X_test, y_test)]
        )



    def predict(self, df):
        """
        """
        df = self.preprocess(df)

        if('y' in df.columns):
            df = df.drop('y', axis=1, inplace=False)

        features = df[self.features].copy()

        out_df = self.model.predict(features)


        return out_df
    


df = ml.data.get_training_flattened()


y_pred = {}

for location in ['A', 'B', 'C']:

    df_location = df[df['location'] == location]

    xgbh = XGBoostHenrik()
    xgbh.train(df)
    y_pred[location] = xgbh.predict(df_location)

print(y_pred)
    
