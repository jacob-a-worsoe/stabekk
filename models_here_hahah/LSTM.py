
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# In module1.py
import sys
import os

from sklearn.metrics import mean_absolute_error

# Get the absolute path of folder2
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, os.pardir)
folder2_dir = os.path.join(parent_dir, 'ml_combat')

# Add folder2 to sys.path
sys.path.append(parent_dir)

from ml_combat.MetaModel import MetaModel
import ml_combat as ml
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import timeseries_dataset_from_array
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

class LSTM(MetaModel):
    def __init__(self, window_len):
        super().__init__("LSTM")

        self.keep_columns = ['total_rad_1h:J',
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

        self.window_len = window_len
        self.batch_size = 10
        self.num_features = None
        self.epochs = 1
        self.scaler = None
        self.num_features_history = []
    
    def test(self, df, n_splits=5):
        print(f"Testing {self.model_name}")
        column_names = df.columns.tolist()
        if 'y' not in column_names:
            raise Exception(f"Missing observed y in columns. Available are {column_names}")

        # This is unecessary because we already clean it when calling train
        #df_cleaned = self.preprocess(df)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        MAE_values = []

        for train_index, test_index in tscv.split(df):

            train_partition = df.iloc[train_index]
            valid_partition = df.iloc[test_index]

            # Needs to be adjusted due to window length
            adj_valid_partition = pd.concat([train_partition[-self.window_len:], valid_partition], ignore_index=True)

            self.train(train_partition)
            predictions = self.predict(adj_valid_partition)
            
            y_true = valid_partition['y']
            y_pred = predictions['y_pred']

            MAE = mean_absolute_error(y_true, y_pred)
            MAE_values.append(MAE)

            print(f'Train-Test ratio:{len(train_partition)/len(valid_partition)} achieved MAE {MAE}')

        return MAE_values
        
    def preprocess(self, df, has_target_col=False):
        temp_df = df.copy()

        temp_df['total_rad_1h:J'] = df['diffuse_rad_1h:J'] + df['direct_rad_1h:J']

        # Only keep the columns in keep_columns that df actually has
        keep_columns_exist = [col for col in self.keep_columns if col in temp_df.columns]
        temp_df = temp_df[keep_columns_exist]
        
        # Clean out NaNs/NONEs -- May need to remove for more than just total_rad_1h:J
        #temp_df['total_rad_1h:J'].fillna(df['total_rad_1h:J'].interpolate().cummax())
        #temp_df = temp_df.dropna(axis=0, how="all", subset="total_rad_1h:J")

        if(has_target_col):
            temp_df['y'].fillna(df['y'].interpolate().cummax())
            # temp_df = temp_df.dropna(axis=0, how="all", subset="y")

        # Min-max scale all columns so all values in [0, 1]
        self.scaler = StandardScaler()
        temp_np = self.scaler.fit_transform(temp_df)

        # Convert to time-series of given window-lengths 
        # (Sequence_length is how long each time-window is, sequence_stride is how long the window shifts forward in time for each sequence)
        if(has_target_col):
            self.num_features = temp_np.shape[1] - 1

            ## TEMPORARY ##
            self.num_features_history.append(self.num_features)
            ###############

            features = temp_np[:, : self.num_features]
            targets = temp_np[:, self.num_features]

            temp_df = TimeseriesGenerator(data=features, targets=targets, length=self.window_len, shuffle=False, batch_size=self.batch_size)
        
        else:
            self.num_features = temp_np.shape[1]
            
            ## TEMPORARY ##
            self.num_features_history.append(self.num_features)
            ###############

            features = temp_np[:, : self.num_features]
            dummy_targets = np.zeros(features.shape)

            temp_df = TimeseriesGenerator(data=features, targets=dummy_targets, length=self.window_len, shuffle=False, batch_size=self.batch_size)
        
        # Returns a generator (keras datatype)
        return temp_df

    def train(self, df):
        # Make train-test split
        train_end = round(len(df) * 0.7)
        # test_df = df.iloc[train_end:]

        train_generator = self.preprocess(df, has_target_col = True)
        print("20*12")
        # test_generator = self.preprocess(test_df, has_target_col=True)

        # Design the model
        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.LSTM(64, input_shape=(self.window_len, self.num_features), return_sequences=True))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.5))

        self.model.add(tf.keras.layers.LSTM(64, return_sequences=True))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.5))

        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.LSTM(32, return_sequences=False))

        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(1))

        # Add pruning measures
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
        self.model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=tf.metrics.MeanAbsoluteError())

        # Train the model
        history = self.model.fit(train_generator, epochs=self.epochs, shuffle=False, callbacks=[early_stopping])


    def predict(self, df: pd.DataFrame):
        """
        """
        output_generator = self.preprocess(df, ('y' in df.columns))
        model_out = self.model.predict(output_generator)

        print(model_out)
        print(model_out.shape)

        ### RESCALE BACK TO ORIGINAL SIZE
        temp_scaler = StandardScaler()
        index_of_y = self.keep_columns.index("y")
        temp_scaler.mean_, temp_scaler.scale_ = self.scaler.mean_[index_of_y], self.scaler.scale_[index_of_y]

        scaled_model_out = temp_scaler.inverse_transform(model_out)
        print(scaled_model_out)

        data = {'y_pred': scaled_model_out.flatten()}
        out_df = pd.DataFrame(data)

        if(len(df) != len(out_df) + self.window_len):
            print("INPUT DF WAS CLEANED S.T. INDEXES WONT MATCH UP")
            print("Original df:", len(df), "Output pred_df", len(out_df))

        return out_df
    


df_train = ml.data.get_training_flattened()
df_test = ml.data.get_testing_flattened()
window_len = 50

for location in ['A', 'B', 'C']:
    temp_df = df_train[df_train['location']==location]

    lstm = LSTM(window_len=window_len)
    lstm.test(temp_df)

