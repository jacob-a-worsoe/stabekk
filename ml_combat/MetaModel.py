
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import statistics

class MetaModel(ABC):     
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

        return

    # Denne df-en må ha y som kolonne
    def test(self, df: pd.DataFrame, n_splits=5):
        """
            Expanding window cross-validation, df must have y in it for testing against predictions
        """
        print(f"Testing {self.model_name}")
        column_names = df.columns.tolist()
        if 'y' not in column_names:
            raise Exception(f"Missing observed y in columns. Available are {column_names}")

        # This is unecessary because we already clean it when calling train
        # drop_y_with_na
        df = df.dropna(subset=['y'], inplace=False)

        MAE_values = []
        MSE_values = []

        # tscv = TimeSeriesSplit(n_splits=n_splits)
        kf =KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(df):
            train_partition = df.iloc[train_index]
            valid_partition = df.iloc[test_index]

            self.train(train_partition)
            predictions = self.predict(valid_partition)
            
            y_true = valid_partition['y']
            y_pred = predictions['y_pred']

            MAE = mean_absolute_error(y_true, y_pred)
            MAE_values.append(MAE)

            MSE_values.append((y_pred - y_true).mean())

            print(f"Run {len(MAE_values)} MAE =", MAE)

        print("Mean Signed Error vals", MSE_values)
        average_mae = statistics.mean(MAE_values)
        print("MAE Vals: MEAN:", average_mae, 'ALL:' , MAE_values)
        
        return MAE_values
    
    @abstractmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """
            Takes in single-index (datetime as index) df, and returns df with only desired features
        """    
        pass
    
    @abstractmethod
    def train(df: pd.DataFrame):
        """

        """
        pass

    @abstractmethod
    def predict(df: pd.DataFrame):
        """
            Runs trained model on on input df, preprocessing the df first and then returns datetime and y_pred
        """
        pass