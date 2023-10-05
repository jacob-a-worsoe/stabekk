"""
Contains the following functions:

    * get_training - gets full training data (duh)
    * get_test - gets the feature estimates used for the forecast
"""

import pandas as pd

from . import module_dir

data_dir = module_dir + '/../data/'


def get_training():
    """ gets full training data (duh) """

    train_a = pd.read_parquet(data_dir + 'A/train_targets.parquet')
    train_b = pd.read_parquet(data_dir + 'B/train_targets.parquet')
    train_c = pd.read_parquet(data_dir + 'C/train_targets.parquet')

    X_train_estimated_a = pd.read_parquet(data_dir + 'A/X_train_estimated.parquet')
    X_train_estimated_b = pd.read_parquet(data_dir + 'B/X_train_estimated.parquet')
    X_train_estimated_c = pd.read_parquet(data_dir + 'C/X_train_estimated.parquet')

    X_train_observed_a = pd.read_parquet(data_dir + 'A/X_train_observed.parquet')
    X_train_observed_b = pd.read_parquet(data_dir + 'B/X_train_observed.parquet')
    X_train_observed_c = pd.read_parquet(data_dir + 'C/X_train_observed.parquet')


def get_testing():
    """ gets the feature estimates used for the forecast """

    X_test_estimated_a = pd.read_parquet(data_dir + 'A/X_test_estimated.parquet')
    X_test_estimated_b = pd.read_parquet(data_dir + 'B/X_test_estimated.parquet')
    X_test_estimated_c = pd.read_parquet(data_dir + 'C/X_test_estimated.parquet')

    X_test_estimated_a.insert(2, 'location', 'A')
    X_test_estimated_b.insert(2, 'location', 'B')
    X_test_estimated_c.insert(2, 'location', 'C')

    return pd.concat([X_test_estimated_a, X_test_estimated_b, X_test_estimated_c])
