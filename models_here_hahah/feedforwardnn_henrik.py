
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

from keras import models, layers, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf


class FeedforwardNNHenrik(MetaModel):
    
    def __init__(self):
        super().__init__("Feedforward Neural Network Henrik")
        self.features = []

        """
        self.features.extend(['month',
                             'hour',
                            'total_rad_1h:J',
                            'sun_elevation:d'])
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

        # Make NaN interpolated
        ml.utils.interpolate_na(temp_df, self.features)


        # Normalize the features
        scaler = preprocessing.MinMaxScaler()
        temp_df[self.features] = scaler.fit_transform(temp_df[self.features])

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

        
        print('Num features:', len(self.features))

        # Setup XGB
        self.model = models.Sequential()
        """
        self.model.add(layers.Dense(input_dim=len(self.features),
                       units=len(self.features), 
                       activation="relu"))
        
        self.model.add(layers.Dense(input_dim=len(self.features),
                       units=32, 
                       activation="relu"))
        
        self.model.add(layers.Dense(input_dim=32,
                       units=32, 
                       activation="relu"))


        self.model.add(layers.Dense(input_dim=32,
                       units=32, 
                       activation="relu"))
        
        self.model.add(layers.Dense(input_dim=32,
                       units=32, 
                       activation="relu"))
        

        self.model.add(layers.Dense(input_dim=32,
                       units=16, 
                       activation="relu"))
        """

        # Params
        learning_rate = 0.5
        

        # add the output layer
        self.model.add(layers.Dense(input_dim=len(self.features),
                            units=1,
                            activation='relu'))

        # define our loss function and optimizer
        self.model.compile(loss='mean_absolute_error',
                    # Adam is a kind of gradient descent
                    optimizer=optimizers.SGD(learning_rate=learning_rate))


        self.model.fit(
            X,
            y,
            verbose=True,
            epochs=20
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

        # Make y_preds 1D
        y_preds = np.ravel(y_preds)

        out_df = pd.DataFrame(data={'y_pred': y_preds})

        return out_df
    


df = ml.data.get_training_flattened()

for location in ['A', 'B', 'C']:
    print("###########################################")
    print(f"###############  LOCATION {location} ###############")
    print("###########################################")
    df_location = df[df['location'] == location]

    fnnh = FeedforwardNNHenrik()
    fnnh.test(df_location)


# Generate submittable
ml.utils.make_submittable("FeedforwardNeuralNetwork.csv", model=FeedforwardNNHenrik())

    
"""
NEEDS MORE PREPROCESSING ___ REMOVING NONES/NANS


Best so far; 
- all features

"""