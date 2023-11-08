
############# 
# Path fix
import sys
import os
# Get the absolute path of folder2
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, os.pardir)
# Add to sys.path
sys.path.append(parent_dir)
#############

# Temporarily suppress FutureWarning
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

#############
# Needed to suppress prophet output
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)
#############

import pandas as pd
import numpy as np
from prophet import Prophet

from ml_combat.MetaModel import MetaModel
import ml_combat as ml


###### Start Here ######
class ProphetModel(MetaModel):
    """
    Hyperparameters, default in parentheses:

        'C_observed_only' (False) - whether location C should only use observed data
        'features_by_corr0.6' (True) - use features with corr > 0.6 with y, if false uses 'total_rad_1h:J'
    """
    
    def __init__(self, hyperparameters = None):
        super().__init__("Prophet Model")
        self.hyperparameters = hyperparameters
        
    def preprocess(self, df):
        """
        """
        df = df.copy()

        if self.hyperparameters is not None and self.hyperparameters["C_observed_only"]:
            if location == "C":
                df = df[df.weather_data_type == 'observed'].copy()


        df['total_rad_1h:J'] = df['diffuse_rad_1h:J'] + df['direct_rad_1h:J']

        df = df.dropna(axis=0, how="all", subset="total_rad_1h:J")

        if('y' in df.columns.tolist()):
            df = df.dropna(axis=0, how="all", subset="y")

        df.fillna(0, inplace=True)
        
        return df

    def train(self, df):
        """
        """
        df = self.preprocess(df)

        self.prophet_model = Prophet()
        
        feat_find = df.corr().y.apply(abs).sort_values(ascending=False)
        features = feat_find[feat_find > 0.6].index.tolist()
        features.remove("y")
        # features = ['total_rad_1h:J']
        #     'is_in_shadow:idx', 
        #     'is_day:idx',
        #     'sun_elevation:d',
        #     'diffuse_rad_1h:J',
        #     'diffuse_rad:W',
        #     'clear_sky_energy_1h:J',
        #     'clear_sky_rad:W',
        #     'direct_rad_1h:J',
        #     'direct_rad:W'
        # ]
        # for feat in [i for i in df.columns.to_list() if i not in ['location', 'ds', 'y', 'weather_data_type']]:
        
        if self.hyperparameters is not None and not self.hyperparameters["features_by_corr0.6"]:
            features = ['total_rad_1h:J']
        
        for feat in features:
            self.prophet_model.add_regressor(feat)

        self.prophet_model.fit(df)


    def predict(self, df):
        """
        """
        df = self.preprocess(df)

        forecast = self.prophet_model.predict(df)
        # self.prophet_model.plot_components(forecast)
        # fig = prophet_model.plot_components(forecast)
        temp_ret = forecast[['ds', 'yhat']].rename(columns={'yhat':'y_pred'})

        # force negative values to zero
        temp_ret.y_pred = temp_ret.y_pred.apply(lambda a : max(a, 0))

        return temp_ret
    

df = ml.data.get_training_flattened()

for location in ['A', 'B', 'C']:
    temp_df = df[df['location'] == location]

    lr = ProphetModel()
    lr.test(temp_df)

print("Done creating a prophet model!")