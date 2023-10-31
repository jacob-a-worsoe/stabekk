
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

from sklearn.model_selection import train_test_split

from xgboost_henrik import XGBoostHenrik
from catboost_henrik import CatBoostHenrik
from lightgbm_henrik import LightBGMHenrik



class TreeCompositeHenrik(MetaModel):
    
    def __init__(self):
        super().__init__("CatBoost Henrik")
        self.features = []
        
        self.features.extend(['month',
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

        temp_df['total_rad_1h:J'] = temp_df['diffuse_rad_1h:J'] + temp_df['direct_rad_1h:J']    
        
        # Extracting hour-of-day and month, and making them cyclical
        temp_df['hour'] = temp_df['ds'].dt.hour
        ml.utils.map_hour_to_seasonal(temp_df, 'hour')

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

    def train(self, df: MetaModel):
        """
        """

        self.models = {
            "XGBoost Henrik": XGBoostHenrik(),
            "CatBoost Henrik": CatBoostHenrik(),
            "LightGBM Henrik": LightBGMHenrik()
        }

        for key in self.models:
            self.models[key].train(df)



    def predict(self, df):
        """
        """
        
        all_preds = None

        out_df = None

        for key in self.models:
            y_pred = self.models[key].predict(df)['y_pred']
            if(all_preds is None):
                all_preds = pd.DataFrame(y_pred)
            else:
                all_preds[key] = y_pred.values

            
        avg_series = all_preds.mean(axis=1)

        print("The different models produced the following predictions:")
        print(all_preds)
        print("Averages")
        print(avg_series)

        return pd.DataFrame(avg_series, columns=['y_pred'])


df = ml.data.get_training_flattened()

for location in ['A', 'B', 'C']:
    print("###########################################")
    print(f"###############  LOCATION {location} ###############")
    print("###########################################")
    df_location = df[df['location'] == location]

    tch = TreeCompositeHenrik()
    tch.test(df_location)


# Generate submittable
ml.utils.make_submittable("TreeCompositeHenrik.csv", model=TreeCompositeHenrik())

    
"""
Best so far; 
- all features

Location A -- MAE Vals [324.82611381886295, 148.21047087943, 207.51289861988295, 229.10210810565073, 124.80138500375826]
Location B -- MAE Vals [19.43038858877779, 78.7647672132421, 51.03549389364723, 38.44011339001294, 32.82959121516759]
Location C -- MAE Vals [75.26357382878206, 10.401516210882635, 43.9462814687844, 7.503776578680513, 22.991776009994577]

"""