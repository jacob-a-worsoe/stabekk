
import pandas as pd

from . import module_dir
from . import data

def interpolate_na(df: pd.DataFrame, cols: list, inplace=True):
    if inplace:
        for col in cols:
            df[col].ffill(inplace=True)
            df[col].bfill(inplace=True)

            df[col].interpolate(inplace=True)   

            df[col].fillna(df[col].cummax(), inplace=True)

    else:
        new_df = None
        for col in cols:
            new_df = df[col].ffill(inplace=inplace)
            new_df = df[col].bfill(inplace=inplace)

            new_df = df[col].fillna(df[col].interpolate().cummax(), inplace=inplace)

        return new_df


def map_hour_to_seasonal(df, time_col):
    mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
               7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
               13: 11, 14: 10, 15: 9, 16: 8, 17: 7, 18: 6,
               19: 5, 20: 4, 21: 3, 22: 2, 23: 1, 24: 0,}
    df[time_col] = df[time_col].replace(mapping, inplace=True)


# liste med 'y_pred' 

def y_pred_to_csv(file_name, df):
    df[['y_pred']].reset_index(drop=True).reset_index().rename(columns={'index': 'id', 'y_pred': 'prediction'}).to_csv(module_dir + "/../submissions/" + file_name, index=False)

def make_submittable(file_name, model = None, model_dict = None):
    """
    model if same model used for all locations
    model dict if not

    model is instance of model (model = MetaModel())
    model dict uses location as key ({"A": MetaModel(), "B": MetaModel(), "C": MetaModel()})
    """

    df = data.get_training_flattened()
    test = data.get_testing_flattened()
    ret = pd.DataFrame()
    
    for location in ['A', 'B', 'C']:
        temp_df = df[df['location']==location]
        temp_test = test[test['location']==location]

        if model is not None:
            m = model
        elif model_dict is not None:
            m = model_dict[location]
        else:
            raise ValueError("no model specified")
    
        m.train(temp_df)
        fcst = m.predict(temp_test)

        ret = pd.concat([ret, fcst])

    y_pred_to_csv(file_name, ret)