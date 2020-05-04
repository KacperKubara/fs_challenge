from typing import List

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, StandardScaler

from runner import Runner 
from config import COLS_NUM, COLS_CAT, COLS_OTHER


class Preprocessor(Runner):
    def __init__(self, cols_num: List, cols_to_impute: List, 
                 path_read_X: str, path_read_y: str, dir_write: str):
        """ Data preprocessing class
        
        Parameters
        ----------
        cols_num: List of numerical col names

        path_read_X: path to transaction_obf datasets

        path_read_y: path to labels_obf dataset

        dir_write: directory path to write results to
        """
        self.cols_num = cols_num
        self.cols_to_impute = cols_to_impute
        self.X = pd.read_csv(path_read_X)
        self.y = pd.read_csv(path_read_y)
        self.dir_write = dir_write
        self.l_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def run(self, label_encode: bool = False , 
            hot_encode: bool = False, normalize: bool = False,
            save: bool = False) -> pd.DataFrame:
        """ Runs chosen preprocessing utilites"""
        # Combines X and y into one dataset
        data = self.combine_data()
        # Imputes missing data
        data = self.impute(data)
        # Creates label encoding of cat cols
        if label_encode is True:
            data = self.label_encode(data)
        # Create one hot encoding of cat columns
        if hot_encode is True:
            data = self.hot_encode(data)
        # Normalizes num columns
        if normalize is True:
            data = self.normalize(data)
        return data

    def combine_data(self) -> pd.DataFrame:
        """ Combines X and y data together by matching event ID"""
        data = self.X.\
            assign(fraud=pd.Series(np.zeros(len(self.X.index))).values)
        data.loc[data["eventId"].\
            isin(self.y["eventId"].unique()), "fraud"] = 1
        return data

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Imputes missing/incorrect values in merchantZip col"""
        if "merchantZip" not in self.cols_to_impute:
            return data
        # Valid zipcode expressed by regex        
        zip_code_regex = "[A-Za-z]{1,}[0-9]{1,}"
        data_mask = \
            ~data["merchantZip"].str.match(zip_code_regex).fillna(False)
        # Converting all missing values in merchantZip into NaN
        data.loc[data_mask, "merchantZip"] = np.nan
        # Impute missing values with a mode of the merchantZip col
        mode = data["merchantZip"].mode().values[0]
        data["merchantZip"].fillna(mode, inplace=True)
        return data
    
    def label_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Label encoding for cat columns"""
        for i in range(len(COLS_CAT)):
            data[COLS_CAT[i]] = self.l_encoder.fit_transform(data[COLS_CAT[i]])
        return data       

    def hot_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """ One hot encoding for cat columns"""
        data = pd.get_dummies(data, columns=COLS_CAT)
        return data

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Normalizes numerical columns"""
        data[COLS_NUM] = self.scaler.fit_transform(data[COLS_NUM])
        return data
