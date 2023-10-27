
import warnings

from sklearn.decomposition import PCA

# Temporarily suppress FutureWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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

class PCARegression(MetaModel):
    
    def __init__(self, pca_dimensions: int = 6):
        super().__init__("PCA Regression")
        self.keep_columns = ['ds',
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
                            'absolute_humidity_2m:gm3',
                            'y'
        ]

        self.pca_dimensions = pca_dimensions
    
    def interpolate_na(self, df, cols):
        for col in cols:
            df[col].ffill(inplace=True)
            df[col].bfill(inplace=True)

            df[col].fillna(df[col].interpolate().cummax(), inplace=True)            
        
    def preprocess(self, df):
        """
        """
        temp_df = df.copy()

        has_target_col = 'y' in temp_df.columns

        temp_df['total_rad_1h:J'] = temp_df['diffuse_rad_1h:J'] + temp_df['direct_rad_1h:J']


        self.keep_columns_exist = [col for col in self.keep_columns if col in temp_df.columns]
        self.interpolate_na(temp_df, self.keep_columns_exist)
        temp_df = temp_df[self.keep_columns_exist]

        # Add hour of day
        temp_df['hour'] = temp_df['ds'].dt.hour
        ml.utils.map_hour_to_seasonal(temp_df, 'hour')
        temp_df = temp_df.drop('ds', axis=1)

        num_features = len(self.keep_columns) - 1

        if(has_target_col):
            temp_df['y'].fillna(df['y'].interpolate().cummax(), inplace=True)

        self.scaler = StandardScaler()
        temp_np = self.scaler.fit_transform(temp_df)
        features = temp_np[:, : num_features]

        pca = PCA(n_components=self.pca_dimensions)
        pca_features = pca.fit_transform(features)

        if(has_target_col):
            target = temp_np[:, num_features]
            pca_w_target = np.column_stack((pca_features, target))

            col_names = [f"PCA{i}" for i in range(self.pca_dimensions)] + ['y']

            temp_df = pd.DataFrame(data=pca_w_target, columns=col_names)
        else:
            col_names = [f"PCA{i}" for i in range(self.pca_dimensions)] + ['y']
            temp_df = pd.DataFrame(data=pca_features, columns=col_names)

        return temp_df

    def train(self, df):
        """
        """
        df = self.preprocess(df)

        PCA_features = df.drop('y', axis=1, inplace=False)

        self.model = LinearRegression()

        self.model.fit(PCA_features, df['y'])


    def predict(self, df):
        df = self.preprocess(df)

        if('y' in df.columns):
            df = df.drop('y', axis=1, inplace=False)

        y_preds = self.model.predict(df)
        y_preds_arr = np.array(y_preds)

        out_df = pd.DataFrame(y_preds_arr, columns=["y_pred"]) 

        # SCALE BACK OUTPUT
        temp_scaler = StandardScaler()
        index_of_y = self.keep_columns.index("y")
        temp_scaler.mean_, temp_scaler.scale_ = self.scaler.mean_[index_of_y], self.scaler.scale_[index_of_y]

        scaled_model_out = temp_scaler.inverse_transform(out_df)
        out_df = pd.DataFrame(data=scaled_model_out, columns=['y_pred'])

        out_df['y_pred'] = out_df['y_pred'].apply(lambda a : max(0, a))

        #print("NUM NEGATIVE OUTPUTS", (out_df['y_pred'] < 0).sum(), "AVERAGE NEG VAL", (out_df[out_df['y_pred'] < 0].mean()))

        return out_df
    


df = ml.data.get_training_flattened()

best_score = {'A': 1000, 'B': 1000, 'C': 1000}
best_index = {'A': 0, 'B': 0, 'C': 0}


for location in ['A', 'B', 'C']:
    print("ATTEMPTING LOCATION", location)

    for d in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:

        temp_df = df[df['location'] == location]
        print("DIMENSIONS:", d)
        pca = PCARegression(pca_dimensions=d)
        test_score = pca.test(temp_df)[-1]
        
        if(test_score < best_score[location]):
            best_index[location] = d


print("BEST NUM DIMENSIONS IS", best_index)
print("Done creating a linear regression model!")