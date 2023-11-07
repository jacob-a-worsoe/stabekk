
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
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, train_test_split


class CatBoostHenrik(MetaModel):
    
    def __init__(self):
        super().__init__("CatBoost Henrik")
        self.features = []
        self.features.extend(['sample_importance',
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
            'objective': "MAE",
            'eta': 0.08,
            'iterations': 2000,
            'logging_level': 'Silent'
        }

        
        # Setup XGB
        self.model = cb.CatBoostRegressor(**params)

        """
        print("PERFORMING GRID SEARCH EEEEEEEE")
        # Defining your search space
        hyperparameter_space = {'iterations': [500, 1000, 2000, 3000],
                                'eta': [0.03, 0.7, 0.1, 0.3, 0.8],
                                'depth': [6, 8, 10, 12]}
                            
        cv = KFold(n_splits=5, shuffle=True)
                
        reg = GridSearchCV(self.model, hyperparameter_space, 
                        scoring = 'neg_mean_absolute_error', cv=cv,
                        n_jobs = -1, refit = True)


        reg.fit(
            X,
            y,
            verbose=True,
            sample_weight=temp_df['sample_importance']
        )
        """

        self.model.fit(
            X,
            y,
            verbose=True,
            sample_weight=temp_df['sample_importance']
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
df = ml.data.get_training_cleaned()

for location in ['A', 'B', 'C']:
    print("###########################################")
    print(f"###############  LOCATION {location} ###############")
    print("###########################################")
    df_location = df[df['location'] == location]

    cbh = CatBoostHenrik()
    cbh.train(df_location)
    cbh.test(df_location)
"""
"""

# Generate submittable
ml.utils.make_submittable("CatBoost.csv", model=CatBoostHenrik())

""" 
"""
Best so far; 
- all features
###########################################
###############  LOCATION A ###############
###########################################
MAE Vals: MEAN: 176.14066773038115 ALL: [181.97487500981785, 176.85836905536448, 171.72775237238918, 179.38213214944784, 170.76021006488642]

###########################################
###############  LOCATION B ###############
###########################################
MAE Vals: MEAN: 25.223891878849738 ALL: [25.700806897777664, 26.695572499908113, 23.23840744259385, 24.533980368618945, 25.95069218535011]

###########################################
###############  LOCATION C ###############
###########################################
MAE Vals: MEAN: 19.864001427699083 ALL: [19.638891777935232, 20.648785428223455, 19.68583669963377, 20.02265140733278, 19.32384182537018]

------- OLD -------- (BEFORE CHANGING TEST FUNCTION)
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