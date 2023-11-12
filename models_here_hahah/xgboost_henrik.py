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
    
    def __init__(self, features = None):
        super().__init__("XGBoost Henrik")
        self.features = []
        if features is not None:
            self.features = features  # Use the provided features if not None
        else:
            self.features.extend(['is_estimated', 'sample_importance',
                              'dayofyear',
                             'hour',
                            'total_rad_1h:J',
        'absolute_humidity_2m:gm3',
       'air_density_2m:kgm3', 'ceiling_height_agl:m', 'clear_sky_energy_1h:J',
       'clear_sky_rad:W', 'cloud_base_agl:m', 'dew_or_rime:idx',
       'dew_point_2m:K', 'effective_cloud_cover:p', 'elevation:m',
       'fresh_snow_12h:cm', 'fresh_snow_1h:cm', 'fresh_snow_24h:cm',
       'fresh_snow_3h:cm', 'fresh_snow_6h:cm',
       'is_in_shadow:idx', 'msl_pressure:hPa', 'precip_5min:mm',
       'precip_type_5min:idx', 'pressure_100m:hPa', 'pressure_50m:hPa',
       'prob_rime:p', 'rain_water:kgm2', 'relative_humidity_1000hPa:p',
       'sfc_pressure:hPa', 'snow_density:kgm3', 'snow_depth:cm',
       'snow_drift:idx', 'snow_melt_10min:mm', 'snow_water:kgm2',
       'sun_azimuth:d', 'sun_elevation:d', 'super_cooled_liquid_water:kgm2',
       't_1000hPa:K', 'total_cloud_cover:p', 'visibility:m',
       'wind_speed_10m:ms', 'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms',
       'wind_speed_w_1000hPa:ms'])
            
        
    def preprocess(self, df: pd.DataFrame):
        """
        """
        temp_df = df.copy()

        has_target = 'y' in temp_df.columns        
        
        ##################################################################################### 
        # FEATURE ENGINEERING
        #####################################################################################


         # Emphasize test start-end: Starting date: 2023-05-01 00:00:00 Ending data 2023-07-03 23:00:00
        temp_df['sample_importance'] = 1
        temp_df.loc[(temp_df['ds'].dt.month >= 5) & 
                    (temp_df['ds'].dt.month < 7), 'sample_importance'] = 2
        
        temp_df.loc[(temp_df['ds'].dt.month == 7) &
                    (temp_df['ds'].dt.day <= 4), 'sample_importance'] = 2
        
        # Add is_estimated parameter
        temp_df['is_estimated'] = (temp_df['weather_data_type'] == 'estimated')
        temp_df['is_estimated'] = temp_df['is_estimated'].astype(int)

        temp_df['total_rad_1h:J'] = temp_df['diffuse_rad_1h:J'] + temp_df['direct_rad_1h:J']    
        
        # Extracting hour-of-day and month, and making them cyclical
        temp_df['hour'] = temp_df['ds'].dt.hour
        temp_df['hour'] = (np.sin(2 * np.pi * (temp_df['hour'] - 4)/ 24) + 1) / 2

        temp_df['month'] = temp_df['ds'].dt.month
        temp_df['month'] = (np.sin(2 * np.pi * (temp_df['month'])/ 12) + 1) / 2


        temp_df['dayofyear'] = temp_df['ds'].dt.day_of_year
        temp_df['dayofyear'] = np.sin(2 * np.pi * (temp_df['dayofyear'] - 80)/ 365)
   
        # SETTING NAN TO 0 CONFORMING TO XGBOOST
        temp_df.fillna(0, inplace=True)

        #####################################################################################

        # DROPPING UNEEEDED FEATURES
        if(has_target):
            features_w_y = self.features + ['y']
            temp_df = temp_df[features_w_y]

        else:
            temp_df = temp_df[self.features]

        return temp_df

    def train(self, df):
        """
        """

        temp_df = self.preprocess(df)

        # Separate features and target
        X = temp_df.drop('y', axis=1, inplace=False).copy().values
        y = temp_df['y'].copy().values

        # Train test split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

        params = {
            'objective': "reg:absoluteerror",
            'eta': 0.15,
            'max_depth': 18,
            'lambda': 4.5,
            'iterations': 6000
        }

        # Setup XGB
        self.model = xgb.XGBRegressor(**params)

        self.model.fit(
            X,
            y,
            verbose=True,
        )



    def predict(self, df):
        """
        """
        df = self.preprocess(df)

        features = [col for col in df.columns if col != 'y']
        X = df[features].values
        y_preds = self.model.predict(X)

        # Set all negative predictions to 0
        y_preds = np.maximum(y_preds, 0)

        out_df = pd.DataFrame(data={'y_pred': y_preds})

        return out_df
    

"""
df = ml.data.get_training_flattened()

for location in ['A', 'B', 'C']:
    print("###########################################")
    print(f"###############  LOCATION {location} ###############")
    print("###########################################")
    df_location = df[df['location'] == location]

    xgbh = XGBoostHenrik()
    xgbh.test(df_location)

"""
"""
# Generate submittable
ml.utils.make_submittable("XGBoostHenrik.csv", model=XGBoostHenrik())

"""
    
"""
Best so far; 
- all features
TUNING XGBOOST: https://github.com/jeffheaton/jh-kaggle-util

params = {
    'objective': "reg:absoluteerror",
    'eta': 0.25,
    'max_depth': 7 (greater than this increased error for all locations)
}

###########################################
###############  LOCATION A ###############
###########################################
Testing XGBoost Henrik
MAE Vals [354.4295288609099, 153.00990366393563, 212.35652953343995, 237.27280511309309, 128.45569111321652]
###########################################
###############  LOCATION B ###############
###########################################
Testing XGBoost Henrik
MAE Vals [20.153800356330613, 81.41714111058613, 53.564878963885704, 40.62891133141144, 35.741791660790895]
###########################################
###############  LOCATION C ###############
###########################################
Testing XGBoost Henrik
MAE Vals [88.86359290444013, 12.891829464296443, 47.035847995773594, 7.747385192897654, 24.51771199369076]

"""