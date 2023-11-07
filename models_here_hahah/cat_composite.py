
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
            self.models[f'CATBOOST_{i}'] = CatCompositeHenrik(features = features[i])

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
    cch = CatCompositeHenrik()
    cch.test(df_location)



"""

# Generate submittable
ml.utils.make_submittable("LGBMComposite.csv", model=LGBMCompositeHenrik())
"""
    
"""
Best so far; 
- all features

LATEST RUN
A - MAE Vals: MEAN: 183.05233386089677 ALL: [183.18792470845497, 188.75977707012484, 179.87072071664326, 179.51379971298303, 183.9294470962777]
B - MAE Vals: MEAN: 25.807783651301374 ALL: [25.97709097110887, 25.51560936122758, 26.010697750587987, 25.61228720891779, 25.92323296466464]
C - MAE Vals: MEAN: 20.47067950351959 ALL: [19.930035757890437, 20.656757771403267, 20.716617731720987, 21.041791948722988, 20.00819430786028]

Location A -- MAE Vals [324.82611381886295, 148.21047087943, 207.51289861988295, 229.10210810565073, 124.80138500375826]
Location B -- MAE Vals [19.43038858877779, 78.7647672132421, 51.03549389364723, 38.44011339001294, 32.82959121516759]
Location C -- MAE Vals [75.26357382878206, 10.401516210882635, 43.9462814687844, 7.503776578680513, 22.991776009994577]

"""