
# Standard imports and settings
if True:
    import warnings

    # Temporarily suppress FutureWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

    import pandas as pd
    import numpy as np
    import random
    import math

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

from lightgbm_henrik import LightBGMHenrik


class LGBMCompositeHenrik(MetaModel):
    
    def __init__(self):
        super().__init__("TreeComposite Henrik")

        self.common_features = ['dayofyear',
                             'hour',
                            'total_rad_1h:J',
                            'sun_elevation:d',
                            'sun_azimuth:d',
                            'is_in_shadow:idx',
                            'effective_cloud_cover:p']
        
        self.random_features = ['absolute_humidity_2m:gm3',
                            'air_density_2m:kgm3', 'ceiling_height_agl:m', 'clear_sky_energy_1h:J',
                            'clear_sky_rad:W', 'cloud_base_agl:m', 'dew_or_rime:idx',
                            'dew_point_2m:K', 'elevation:m',
                            'fresh_snow_12h:cm', 'fresh_snow_1h:cm', 'fresh_snow_24h:cm',
                            'fresh_snow_3h:cm', 'fresh_snow_6h:cm', 'msl_pressure:hPa', 'precip_5min:mm',
                            'precip_type_5min:idx', 'pressure_100m:hPa', 'pressure_50m:hPa',
                            'prob_rime:p', 'rain_water:kgm2', 'relative_humidity_1000hPa:p',
                            'sfc_pressure:hPa', 'snow_density:kgm3', 'snow_depth:cm',
                            'snow_drift:idx', 'snow_melt_10min:mm', 'snow_water:kgm2', 'super_cooled_liquid_water:kgm2',
                            't_1000hPa:K', 'total_cloud_cover:p', 'visibility:m',
                            'wind_speed_10m:ms', 'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms',
                            'wind_speed_w_1000hPa:ms']
        
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
        
    def preprocess(self, df: pd.DataFrame):

        return df.copy()

    def train(self, df: MetaModel):
        
        num_models = 10
        #num_rand_features = round(len(self.random_features) * 0.6)
        num_rand_features = round(len(self.random_features) * 1)

        features = dict()

        self.models = dict()

        for i in range(num_models):
            temp_rand_features = random.sample(self.random_features, num_rand_features)

            features[i] = self.common_features + temp_rand_features
            print(f"NUMBER OF FEATURES OF MODEL {i} IS:", len(features[i]))

            self.models[f'LGBM_{i} Henrik'] = LightBGMHenrik(features = features[i])

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

        avg_series = np.maximum(avg_series, 0)

        return pd.DataFrame(avg_series, columns=['y_pred'])


df = ml.data.get_training_cleaned()

for location in ['A', 'B', 'C']:
    print("###########################################")
    print(f"###############  LOCATION {location} ###############")
    print("###########################################")
    df_location = df[df['location'] == location]

    lgbmch = LGBMCompositeHenrik()
    lgbmch.test(df_location)


# Generate submittable
ml.utils.make_submittable("LGBMComposite.csv", model=LGBMCompositeHenrik())

    
"""
Best so far; 
- all features

Location A -- MAE Vals [324.82611381886295, 148.21047087943, 207.51289861988295, 229.10210810565073, 124.80138500375826]
Location B -- MAE Vals [19.43038858877779, 78.7647672132421, 51.03549389364723, 38.44011339001294, 32.82959121516759]
Location C -- MAE Vals [75.26357382878206, 10.401516210882635, 43.9462814687844, 7.503776578680513, 22.991776009994577]

"""