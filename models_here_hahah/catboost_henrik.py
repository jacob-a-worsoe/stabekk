
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

import random
import catboost as cb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split


class CatBoostHenrik(MetaModel):
    
    def __init__(self):
        super().__init__("CatBoost Henrik")
        self.features = []
        
        self.features.extend(['month',
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
        
        """

        self.features.extend(['total_rad_1h:J', 'month', 'hour', 'sun_elevation:d', 'effective_cloud_cover:p'])
        """

        """
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
        """
        """
        self.features.extend(['super_cooled_liquid_water:kgm2',
                              'effective_cloud_cover:p', 'elevation:m',
                              'fresh_snow_1h:cm', 'fresh_snow_24h:cm',
                              'msl_pressure:hPa', 'precip_5min:mm', 'prob_rime:p', 
                              'relative_humidity_1000hPa:p', 'visibility:m'
                              ])
        """                              
        
    def test(self, df: pd.DataFrame, n_splits=5):
        """
            Expanding window cross-validation, df must have y in it for testing against predictions
        """
        print(f"Testing {self.model_name}")
        column_names = df.columns.tolist()
        if 'y' not in column_names:
            raise Exception(f"Missing observed y in columns. Available are {column_names}")

        # This is unecessary because we already clean it when calling train
        # drop_y_with_na
        df = df.dropna(subset=['y'], inplace=False)

        MAE_values = []

        tscv = TimeSeriesSplit(n_splits=n_splits)

        

        for train_index, test_index in tscv.split(df):
            train_partition = df.iloc[train_index]
            valid_partition = df.iloc[test_index]

            self.train(train_partition)
            predictions = self.predict(valid_partition)
            
            y_true = valid_partition['y']
            y_pred = predictions['y_pred']

            MAE = mean_absolute_error(y_true, y_pred)
            MAE_values.append(MAE)

        #y_true_and_pred = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred})
        #y_true_and_pred.to_csv(f"CatBoost_y_true_and_pred_{random.randint(1,10000)}.csv")

        print("MAE Vals", MAE_values)
        
        return MAE_values

    def preprocess(self, df: pd.DataFrame):
        """
        """
        temp_df = df.copy()

        has_target = 'y' in temp_df.columns        
        
        ##################################################################################### 
        # FEATURE ENGINEERING
        #####################################################################################

        temp_df['total_rad_1h:J'] = temp_df['diffuse_rad_1h:J'] + temp_df['direct_rad_1h:J']    
        
        # Extracting hour-of-day and month, and making them cyclical
        temp_df['hour'] = temp_df['ds'].dt.hour
        temp_df['hour'] = (np.sin(2 * np.pi * (temp_df['hour'] - 4)/ 24) + 1) / 2

        
        temp_df['dayofyear'] = temp_df['ds'].dt.day_of_year
        temp_df['dayofyear'] = np.sin(2 * np.pi * (temp_df['dayofyear'] - 80)/ 365)

        temp_df['month'] = temp_df['ds'].dt.month
        ml.utils.map_month_to_seasonal(temp_df, 'month')
   
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
            'objective': "MAE",
            'eta': 0.25,
            'logging_level': 'Silent'
        }


        # Setup XGB
        self.model = cb.CatBoostRegressor(**params)

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

    cbh = CatBoostHenrik()
    cbh.test(df_location)



# Generate submittable
ml.utils.make_submittable("CatBoost.csv", model=CatBoostHenrik())

""" 
"""
Best so far; 
- all features

###########################################
###############  LOCATION A ###############
###########################################
Testing CatBoost Henrik
MAE Vals [360.3682976957419, 162.46404862165602, 224.82038847064652, 240.73085583394803, 133.95529661199055]
###########################################
###############  LOCATION B ###############
###########################################
Testing CatBoost Henrik
MAE Vals [23.139720348715645, 83.8834084925259, 52.620006354026664, 41.204193344231335, 31.86188058508645]
###########################################
###############  LOCATION C ###############
###########################################
Testing CatBoost Henrik
MAE Vals [85.44874912558969, 11.453074841745511, 45.89275838381141, 8.467740475742827, 23.617592499217523]

"""