
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

from autogluon.tabular import TabularDataset, TabularPredictor



class AutoGluonHenrik(MetaModel):
    
    def __init__(self, time_limit=None, excluded_models: list = []):
        super().__init__("AutoGluon Henrik")

        self.time_limit = time_limit
        self.excluded_models = excluded_models

        self.features = []
        
        self.features.extend(['sample_importance',
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

        has_target = 'y' in df.columns

        if has_target:
            temp_df = temp_df[temp_df['y'].notna()]

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
        if(has_target):
            
            return temp_df[self.features + ['y']]
        else:
            return temp_df[self.features]

    def train(self, df):
        """
        """
        temp_df = self.preprocess(df)


        train_data = TabularDataset(temp_df)

        self.model = TabularPredictor(
            label='y',
            eval_metric='mean_absolute_error',
            problem_type='regression',
            sample_weight='sample_importance',
            weight_evaluation=True
        ).fit(train_data,
              time_limit = self.time_limit,
              excluded_model_types = self.excluded_models,
              presets=['good_quality']
              )


    def predict(self, df):
        """
        """
        df = self.preprocess(df)

        features = [col for col in df.columns if col != 'y']
        X = df[features]

        y_preds = self.model.predict(X)
        print("AUTOGLUON MODEL OVERVIEW:")
        print(self.model.get_model_names())
       

        out_df = pd.DataFrame(data={'y_pred': y_preds})

        return out_df
    

"""
df = ml.data.get_training_cleaned()

for location in ['A', 'B', 'C']:
    print("###########################################")
    print(f"###############  LOCATION {location} ###############")
    print("###########################################")
    df_location = df[df['location'] == location]

    agh = AutoGluonHenrik(time_limit=60*1)
    agh.test(df_location)

"""
"""
# Generate submittable
ml.utils.make_submittable("AutoGluon_w_sample_importance_1min.csv", model=AutoGluonHenrik(time_limit=60*1))
"""

    
"""
Best so far; 
- all features

W/O Datetime (probably less overfitted)
A - MAE Vals: MEAN: 164.19447928431063 ALL: [160.86187983685323, 165.58601980572905, 170.3468009073169, 164.7286577981336, 159.4490380735203]
B - MAE Vals: MEAN: 23.00943610618872 ALL: [23.31679828148042, 22.491542616816773, 23.372812317684765, 23.22231308753279, 22.643714227428845]
C - MAE Vals: MEAN: 18.121459098146975 ALL: [18.592314607515814, 18.058354775096685, 18.26069749529676, 17.75047525186657, 17.945453360959053]

With Datetime
A - MAE Vals: MEAN: 149.76513207706887 ALL: [146.83345324318742, 146.6188566849808, 147.27326741611125, 149.8352757089088, 158.26480733215612]
B - MAE Vals: MEAN: 19.99889713887556 ALL: [19.9848420222646, 19.571956697739072, 20.345230150391874, 20.21669685939001, 19.875759964592234]
C - MAE Vals: MEAN: 16.12927055763751 ALL: [16.667676208910606, 16.15111281745623, 16.23317804404866, 15.891535578274334, 15.702850139497727]


FJERNE VINTER alt mellom august og april -- prioritere å overfitte til testsettset
FJERNE Tid-greier
Droppe bagging & stacking
"""