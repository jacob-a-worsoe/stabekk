"""
Contains the following functions:

    * get_spreadsheet_cols - returns the column headers of the file
    * main - the main function of the script
"""

import pandas as pd

from . import module_dir

data_dir = module_dir + '/../data/'

def test():
    print(module_dir)



train_a = pd.read_parquet(data_dir + 'A/train_targets.parquet')
train_b = pd.read_parquet(data_dir + 'B/train_targets.parquet')
train_c = pd.read_parquet(data_dir + 'C/train_targets.parquet')


X_train_estimated_a = pd.read_parquet(data_dir + 'A/X_train_estimated.parquet')
X_train_estimated_b = pd.read_parquet(data_dir + 'B/X_train_estimated.parquet')
X_train_estimated_c = pd.read_parquet(data_dir + 'C/X_train_estimated.parquet')


X_train_observed_a = pd.read_parquet(data_dir + 'A/X_train_observed.parquet')
X_train_observed_b = pd.read_parquet(data_dir + 'B/X_train_observed.parquet')
X_train_observed_c = pd.read_parquet(data_dir + 'C/X_train_observed.parquet')


X_test_estimated_a = pd.read_parquet(data_dir + 'A/X_test_estimated.parquet')
X_test_estimated_b = pd.read_parquet(data_dir + 'B/X_test_estimated.parquet')
X_test_estimated_c = pd.read_parquet(data_dir + 'C/X_test_estimated.parquet')
