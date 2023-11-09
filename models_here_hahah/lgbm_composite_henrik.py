
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
    
    def __init__(self, num_models=10):
        super().__init__("LGBMComposite Henrik")

        self.num_models = num_models
        self.common_features = ['sample_importance', 'is_estimated','dayofyear',
                                'is_day:idx',
                             'hour', 'month',
                            'total_rad_1h:J',
                            'sun_elevation:d',
                            'sun_azimuth:d',
                            'is_in_shadow:idx',
                            'effective_cloud_cover:p']
        
        self.random_features = ['absolute_humidity_2m:gm3',
                            'air_density_2m:kgm3', 'ceiling_height_agl:m', 'clear_sky_energy_1h:J',
                            'clear_sky_rad:W', 'cloud_base_agl:m', 'dew_or_rime:idx',
                            'dew_point_2m:K',
                            'fresh_snow_12h:cm', 'fresh_snow_1h:cm', 'fresh_snow_24h:cm',
                            'fresh_snow_3h:cm', 'fresh_snow_6h:cm', 'msl_pressure:hPa', 'precip_5min:mm',
                            'precip_type_5min:idx', 'pressure_100m:hPa', 'pressure_50m:hPa',
                            'prob_rime:p', 'rain_water:kgm2', 'relative_humidity_1000hPa:p',
                            'sfc_pressure:hPa', 'snow_depth:cm',
                            'snow_water:kgm2', 'super_cooled_liquid_water:kgm2',
                            't_1000hPa:K', 'total_cloud_cover:p', 'visibility:m',
                            'wind_speed_10m:ms', 'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms']
                           
        
    def preprocess(self, df: pd.DataFrame):

        return df

    def train(self, df: pd.DataFrame):
        df = self.preprocess(df)
        num_rand_features = round(len(self.random_features) * 1)

        features = dict()

        self.models = dict()

        for i in range(self.num_models):
            temp_rand_features = random.sample(self.random_features, num_rand_features)
            features[i] = self.common_features + temp_rand_features
            self.models[f'LGBM_{i} Henrik'] = LightBGMHenrik(features = features[i])

            print(f'LGBM_{i} Henrik')

        for key in self.models:
            self.models[key].train(df)



    def predict(self, df):
        """
        """
        
        all_preds = None

        for key in self.models:
            y_pred = self.models[key].predict(df)['y_pred']
            if(all_preds is None):
                all_preds = pd.DataFrame(y_pred)
            else:
                all_preds[key] = y_pred.values

            
        avg_series = all_preds.mean(axis=1)

        avg_series = np.maximum(avg_series, 0)

        return pd.DataFrame(avg_series, columns=['y_pred'])


df = ml.data.get_training_cleaned()

for location in ['A', 'B', 'C']:
    print("###########################################")
    print(f"###############  LOCATION {location} ###############")
    print("###########################################")
    df_location = df[df['location'] == location]

    lgbmch = LGBMCompositeHenrik(num_models=10)
    lgbmch.test(df_location)


# Generate submittable
ml.utils.make_submittable("LGBMComp_new.csv", model=LGBMCompositeHenrik(20))
 
"""
Best so far; 
- all features

LATEST RUN
A (w/o sample_weight) - MAE Vals: MEAN: 162.33737975461767 ALL: [159.3341315613959, 164.1440029369985, 162.39070944464194, 159.8157484296945, 166.00230640035755]
A (w/  sample_weight) - MAE Vals: MEAN: 170.7231177136227 ALL: [167.22454313865978, 172.91097788580325, 171.65778210830626, 167.5210570353448, 174.30122839999945]


A - MAE Vals: MEAN: 183.05233386089677 ALL: [183.18792470845497, 188.75977707012484, 179.87072071664326, 179.51379971298303, 183.9294470962777]
B - MAE Vals: MEAN: 25.807783651301374 ALL: [25.97709097110887, 25.51560936122758, 26.010697750587987, 25.61228720891779, 25.92323296466464]
C - MAE Vals: MEAN: 20.47067950351959 ALL: [19.930035757890437, 20.656757771403267, 20.716617731720987, 21.041791948722988, 20.00819430786028]

Location A -- MAE Vals [324.82611381886295, 148.21047087943, 207.51289861988295, 229.10210810565073, 124.80138500375826]
Location B -- MAE Vals [19.43038858877779, 78.7647672132421, 51.03549389364723, 38.44011339001294, 32.82959121516759]
Location C -- MAE Vals [75.26357382878206, 10.401516210882635, 43.9462814687844, 7.503776578680513, 22.991776009994577]

"""