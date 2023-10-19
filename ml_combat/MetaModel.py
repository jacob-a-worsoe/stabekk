
import pandas as pd
from abc import ABC, abstractmethod

class MetaModel(ABC):     
    def __init__(self, model_name):
        self.model_name = model_name
        self.model: object = None
        self.params: dict = dict()
        return


    def test(self, df):
        # TODO: IMPLEMENT
        df_cleaned = self.preprocess(df)
        y_pred = self.predict(df) #NOE SLIKT etc
        pass
    
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
            Runs trained model on on input df, preprocessing the df first and then 
        """
        pass


class LinearRegressionModel(MetaModel):
    
    def __init__(self):
        super().__init__("Linear Regression")
        

    def preprocess(df):
        """
        """
        pass

    def train(df):
        """
        """
        pass

    def predict(df):
        """
        """
        pass


print("Attempting to create a linear regression model")
lr = LinearRegressionModel()
lr.train()
print("Done creating a linear regression model!")