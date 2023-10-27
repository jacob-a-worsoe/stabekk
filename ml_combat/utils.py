
import pandas as pd

from . import module_dir


def interpolate_na(self, df: pd.DataFrame, cols: list):
    for col in cols:
        df[col].ffill(inplace=True)
        df[col].bfill(inplace=True)

        df[col].fillna(df[col].interpolate().cummax(), inplace=True)    


def map_hour_to_seasonal(df, time_col):
    mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
               7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
               13: 11, 14: 10, 15: 9, 16: 8, 17: 7, 18: 6,
               19: 5, 20: 4, 21: 3, 22: 2, 23: 1, 24: 0,}
    df[time_col] = df[time_col].replace(mapping, inplace=True)
