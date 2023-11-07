
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

from sklearn.linear_model import LinearRegression
from keras import models, layers, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf


class XGBoostComposite(MetaModel):
    
    def __init__(self):
        super().__init__("XGBoost Composite")

        self.common_features = ['is_estimated','dayofyear',
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

    
    def train(self, df: pd.DataFrame, use_meta_learner=True):
        num_models = 30
        num_rand_features = round(len(self.random_features) * 0.8)  
        df = df.copy()
        df['month'] = df['ds'].dt.month

        meta_train_df = df[(df['month'] == 5) | (df['month'] == 6) | (df['month'] == 7)].sample(frac=0.5)
        print("Meta-train % of full DF", len(meta_train_df)/len(df))
        train_df = df.loc[~df.index.isin(meta_train_df)]

        features = dict()
        self.models = dict()

        for i in range(num_models):
            temp_rand_features = random.sample(self.random_features, num_rand_features)
            features[i] = self.common_features + temp_rand_features
            self.models[f'XGBOOST_{i}'] = XGBoostHenrik(features = features[i])

        for key in self.models:
            print("Training model", key)
            self.models[key].train(train_df)
        
        if (use_meta_learner):                        
            y_preds = self.predict(meta_train_df, meta_training=True)

            self.meta_learner = LinearRegression(fit_intercept=False, positive=True)
            self.meta_learner.fit(y_preds, meta_train_df['y'])

            # Adjust weights so all are non-zero and positive
            new_coefficients = np.copy(self.meta_learner.coef_) + 1/num_models
            new_coefficients = new_coefficients / new_coefficients.min()
            new_coefficients = new_coefficients / new_coefficients.sum()
            self.meta_learner.coef_ = new_coefficients

            print(self.meta_learner.coef_)
    
    def predict(self, df, meta_training = False):
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

        if (meta_training):
            print("RETURNING ALL_PREDS")
            return pd.DataFrame(all_preds)

        #print("THIS HAS GONE TOO FAR!")

        # Use meta-learner to calculate final output

        out_np = self.meta_learner.predict(all_preds)
        #print(out_np)

        """
        out_np = all_preds.mean(axis=1)

        print("The different models produced the following predictions:")
        print(out_np)

        out_np = np.maximum(out_np, 0)
        """
        return pd.DataFrame(out_np, columns=['y_pred'])

df = ml.data.get_training_cleaned()

for location in ['A', 'B', 'C']:
    print("###########################################")
    print(f"###############  LOCATION {location} ###############")
    print("###########################################")
    df_location = df[df['location'] == location]
    xgbh = XGBoostComposite()
    xgbh.test(df_location)


"""
# Generate submittable
ml.utils.make_submittable("XGBoostComposite_metalearner2_linreg.csv", model=XGBoostComposite())
"""

""" 
XGBoostComposite.csv
Submitted by Simen Burgos · Submitted 36 seconds ago
Score: 153.89125

LATEST RUNS
A - MAE Vals: MEAN: 176.62721808380883 ALL: [168.99877575019693, 180.4480632325727, 178.82757516377688, 176.05084922389838, 178.81082704859918]
B - MAE Vals: MEAN: 24.398035517766242 ALL: [23.2599901705447, 25.173845344778513, 24.416798181148753, 24.993148435509546, 24.1463954568497]
C - MAE Vals: MEAN: 18.86035746641255 ALL: [19.491336821031307, 18.651404001893223, 18.51260557692587, 18.329057939720226, 19.317382992492124]


MAE Vals: MEAN: 176.4710274970065 ALL: [176.41420818126772, 176.8975072766353, 178.2765501915676, 170.0597955739673, 180.70707626159455]
MAE Vals: MEAN: 175.39589141177154 ALL: [178.58500530367314, 175.55516430285064, 179.60437768681572, 169.4643138372933, 173.77059592822494]
Lin.reg MAE Vals: MEAN: 175.72564969509736 ALL: [186.16878994566966, 176.53021014601214, 173.43554724344588, 175.2040667374494, 167.2896344029097]

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