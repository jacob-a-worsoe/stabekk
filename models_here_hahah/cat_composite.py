
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

from sklearn.linear_model import LinearRegression
from catboost_henrik import CatBoostHenrik


class CatCompositeHenrik(MetaModel):
    
    def __init__(self):
        super().__init__("CatComposite Henrik")
        self.common_features = ['sample_importance', 'is_estimated','dayofyear',
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
        num_models = 1
        num_rand_features = round(len(self.random_features) * 0.7)  
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
            self.models[f'CATBOOST_{i}'] = CatBoostHenrik(features = features[i])

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

        # Use meta-learner to calculate final output (DISABLED)
        out_np = self.meta_learner.predict(all_preds)
        #print(out_np)

        """
        out_np = all_preds.mean(axis=1)

        print("The different models produced the following predictions:")
        print(out_np)

        out_np = np.maximum(out_np, 0)
        """

        return pd.DataFrame(out_np, columns=['y_pred'])


"""
df = ml.data.get_training_cleaned()

for location in ['A', 'B', 'C']:
    print("###########################################")
    print(f"###############  LOCATION {location} ###############")
    print("###########################################")
    df_location = df[df['location'] == location]
    cch = CatCompositeHenrik()
    cch.test(df_location)

"""

# Generate submittable
ml.utils.make_submittable("CatComposite_30models.csv", model=CatCompositeHenrik())

    
"""
Best so far; 
- all features

A - MAE Vals: MEAN: 172.99512889193852 ALL: [170.54521130676503, 172.51186547203955, 174.60158893679454, 170.50780789497156, 176.80917084912187]
B - MAE Vals: MEAN: 24.559548635563765 ALL: [25.072575416215365, 24.284983043035556, 24.418842542053806, 24.088704659474185, 24.932637517039904]
C - MAE Vals: MEAN: 19.222937143220545 ALL: [18.699964153906414, 19.994489192852583, 19.932591821491265, 18.982563126163527, 18.505077421688934]
"""