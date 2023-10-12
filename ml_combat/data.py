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

    ret = pd.DataFrame()
    ret.set_index(pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['location', 'datetime']), inplace=True)
    ret.columns = pd.MultiIndex(levels=[[],[], []], codes=[[],[], []], names=['feature_type', 'minutes', 'feature_name'])


    # estimated data
    data_dict = {'A': X_train_estimated_a, 'B': X_train_estimated_b, 'C': X_train_estimated_c}

    for loc in data_dict:
        out_temp = pd.DataFrame(data_dict[loc].date_forecast.dt.floor('H').unique(), columns=['date_forecast'])
        out_temp.set_index('date_forecast', inplace=True)
        out_temp.columns = pd.MultiIndex.from_product([[], [], out_temp.columns], names=['feature_type', 'minutes', 'feature_name'])

        for m in [0, 15, 30, 45]:
            temp = data_dict[loc][data_dict[loc].date_forecast.dt.minute == m].copy()
            temp.set_index(temp.date_forecast.dt.floor('H'), inplace=True)
            temp.drop(columns=['date_calc', 'date_forecast'], inplace=True)
            temp.columns = pd.MultiIndex.from_product([['estimated'], [m], temp.columns], names=['feature_type', 'minutes', 'feature_name'])
            out_temp = out_temp.merge(temp,left_index=True, right_index=True, how='outer')

        out_temp.set_index(pd.MultiIndex.from_product([[loc], out_temp.index], names=['location', 'datetime']), inplace=True)
        ret = pd.concat([ret, out_temp])

    # observed data
    data_dict = {'A': X_train_observed_a, 'B': X_train_observed_b, 'C': X_train_observed_c}

    for loc in data_dict:
        out_temp = pd.DataFrame(data_dict[loc].date_forecast.dt.floor('H').unique(), columns=['date_forecast'])
        out_temp.set_index('date_forecast', inplace=True)
        out_temp.columns = pd.MultiIndex.from_product([[], [], out_temp.columns], names=['feature_type', 'minutes', 'feature_name'])

        for m in [0, 15, 30, 45]:
            temp = data_dict[loc][data_dict[loc].date_forecast.dt.minute == m].copy()
            temp.set_index(temp.date_forecast.dt.floor('H'), inplace=True)
            temp.drop(columns=['date_forecast'], inplace=True)
            temp.columns = pd.MultiIndex.from_product([['observed'], [m], temp.columns], names=['feature_type', 'minutes', 'feature_name'])
            out_temp = out_temp.merge(temp,left_index=True, right_index=True, how='outer')

        out_temp.set_index(pd.MultiIndex.from_product([[loc], out_temp.index], names=['location', 'datetime']), inplace=True)
        ret = pd.concat([ret, out_temp])
        


    # # train data
    data_dict = {'B': train_b, 'C': train_c} # 'A': train_a, 

    out_temp = train_a.dropna().copy()
    out_temp.rename(columns={'time': 'datetime'}, inplace=True)
    out_temp.set_index('datetime', inplace=True)
    out_temp.columns = pd.MultiIndex.from_product([['y'], ['NA'], out_temp.columns], names=['feature_type', 'minutes', 'feature_name'])

    out_temp.set_index(pd.MultiIndex.from_product([['A'], out_temp.index], names=['location', 'datetime']), inplace=True)

    for loc in data_dict:

        out_temp2 = data_dict[loc].dropna().copy()
        out_temp2.rename(columns={'time': 'datetime'}, inplace=True)
        out_temp2.set_index('datetime', inplace=True)
        out_temp2.columns = pd.MultiIndex.from_product([['y'], ['NA'], out_temp2.columns], names=['feature_type', 'minutes', 'feature_name'])

        out_temp2.set_index(pd.MultiIndex.from_product([[loc], out_temp2.index], names=['location', 'datetime']), inplace=True)

        out_temp = pd.concat([out_temp, out_temp2])

        # ret = ret.merge(out_temp, left_index=True, right_index=True, how='outer')
        # ret = pd.concat([ret, out_temp])
        

    ret = pd.merge(out_temp, ret, left_index=True, right_index=True, how='outer')

    return ret


def get_testing():
    """ gets the feature estimates used for the forecast """

    X_test_estimated_a = pd.read_parquet(data_dir + 'A/X_test_estimated.parquet')
    X_test_estimated_b = pd.read_parquet(data_dir + 'B/X_test_estimated.parquet')
    X_test_estimated_c = pd.read_parquet(data_dir + 'C/X_test_estimated.parquet')

    ret = pd.DataFrame()
    ret.set_index(pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['location', 'datetime']), inplace=True)
    ret.columns = pd.MultiIndex(levels=[[],[], []], codes=[[],[], []], names=['feature_type', 'minutes', 'feature_name'])


    # estimated data
    data_dict = {'A': X_test_estimated_a, 'B': X_test_estimated_b, 'C': X_test_estimated_c}

    for loc in data_dict:
        out_temp = pd.DataFrame(data_dict[loc].date_forecast.dt.floor('H').unique(), columns=['date_forecast'])
        out_temp.set_index('date_forecast', inplace=True)
        out_temp.columns = pd.MultiIndex.from_product([[], [], out_temp.columns], names=['feature_type', 'minutes', 'feature_name'])

        for m in [0, 15, 30, 45]:
            temp = data_dict[loc][data_dict[loc].date_forecast.dt.minute == m].copy()
            temp.set_index(temp.date_forecast.dt.floor('H'), inplace=True)
            temp.drop(columns=['date_calc', 'date_forecast'], inplace=True)
            temp.columns = pd.MultiIndex.from_product([['estimated'], [m], temp.columns], names=['feature_type', 'minutes', 'feature_name'])
            out_temp = out_temp.merge(temp,left_index=True, right_index=True, how='outer')

        out_temp.set_index(pd.MultiIndex.from_product([[loc], out_temp.index], names=['location', 'datetime']), inplace=True)
        ret = pd.concat([ret, out_temp])

    return ret
