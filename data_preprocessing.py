from typing import List

import pandas as pd 
import numpy as np 

from runner import Runner 


class Preprocessor(Runner):
    def __init__(self, cols_num: List, path_read_X: str, 
                 path_read_y: str, dir_write: str):
        """ Data preprocessing class
        
        Parameters
        ----------
        cols_num: List of numerical col names

        path_read_X: path to transaction_obf datasets

        path_read_y: path to labels_obf dataset

        dir_write: directory path to write results to
        """
        self.cols_num = cols_num
        self.X = pd.read_csv(path_read_X)
        self.y = pd.read_csv(path_read_y)
        self.dir_write = dir_write

    def run(self, save=False):
        """ Runs chosen preprocessing utilites"""
        self.combine_data()
        self.impute()

    def combine_data(self):
        pass

    def impute(self):
        raise NotImplementedError

    def replace_nans(self):
        raise NotImplementedError
