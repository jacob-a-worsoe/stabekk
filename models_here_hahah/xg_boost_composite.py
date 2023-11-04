
# Standard imports and settings
if True:
    import warnings

    # Temporarily suppress FutureWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

    import pandas as pd
    import numpy as np
    import random

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
    from models_here_hahah.xgboost_henrik import XGBoostHenrik
    import ml_combat as ml

import xgboost as xgb
from sklearn.model_selection import train_test_split


class XGBoostComposite(MetaModel):
    
    def __init__(self):
        super().__init__("XGBoost Composite")

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
                    
        
    def preprocess(self, df: pd.DataFrame):
        return df.copy()

    
    def train(self, df: MetaModel):
        num_models = 10
        num_rand_features = round(len(self.random_features) * 0.8)  

        features = dict()
        self.models = dict()

        for i in range(num_models):
            temp_rand_features = random.sample(self.random_features, num_rand_features)
            features[i] = self.common_features + temp_rand_features
            self.models[f'XGBOOST_{i}'] = XGBoostHenrik(features = features[i])

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

df = ml.data.get_training_flattened()

for location in ['A', 'B', 'C']:
    print("###########################################")
    print(f"###############  LOCATION {location} ###############")
    print("###########################################")
    df_location = df[df['location'] == location]
    xgbh = XGBoostComposite()
    xgbh.test(df_location)

# Generate submittable
ml.utils.make_submittable("XGBoostComposite.csv", model=XGBoostComposite())


""" 
XGBoostComposite.csv
Submitted by Simen Burgos · Submitted 36 seconds ago
Score: 153.89125
MAE Vals [332.47301366935926, 151.45355247654584, 208.1210777367038, 230.58026635990421, 127.19053024488258]
MAE Vals [18.71474734624145, 78.44080479557826, 48.593741592237855, 37.43438953168417, 35.186301673912666]
MAE Vals [124.32228000041772, 11.053873781924862, 44.72883110227043, 7.794294405931378, 22.627332494436406]

-------------------
PROPOSALS FOR IMPROVEMENTS
- DIFFERENT COMMON FEATURES (SEARCH FOR OPTIMAL)
- AVERAGE MORE MODELS -> UTILIZE LLN
- HYPERPARAMETER TUNING: TREE DEPTH ETC. 


-------------------
EXLORATORY TESTING:
10 models, 80% of random features
MAE Vals [342.42880076791033, 150.9735979170676, 207.40805478418304, 233.99909399711794, 127.32527251201326]
MAE Vals [23.391167475806654, 78.42548396780208, 50.228871681750334, 42.75979403715092, 33.802341939254134]
MAE Vals [93.89999334048738, 9.905491758681157, 44.18740176930063, 7.847838274736086, 22.6658833653653]

15 models, 90% of random features
MAE Vals [341.21920988264776, 152.15535612474346, 208.84172371617535, 234.53106927619862, 127.00073540177058]
MAE Vals [20.265579717193262, 78.97389610943122, 50.00055218241649, 43.47363689382275, 34.60860149881338]
MAE Vals [96.84102497175401, 9.569280902050487, 44.40764779615767, 7.789169001344842, 22.705647534131696]

SINGULAR XG_BOOST
MAE Vals [354.4295288609099, 153.00990366393563, 212.35652953343995, 237.27280511309309, 128.45569111321652]
MAE Vals [20.153800356330613, 81.41714111058613, 53.564878963885704, 40.62891133141144, 35.741791660790895]
MAE Vals [88.86359290444013, 12.891829464296443, 47.035847995773594, 7.747385192897654, 24.51771199369076]
"""