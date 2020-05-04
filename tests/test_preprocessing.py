import unittest

from data_preprocessing import Preprocessor
from config import COLS_NUM, PATH_READ_X, PATH_READ_y, DIR_EDA


class TestPreprocessor(unittest.TestCase):
    """ Class for testing EDA class"""
    def test_compile(self):
        preprocessor = Preprocessor(COLS_NUM, ["merchantZip"], 
                                    PATH_READ_X, PATH_READ_y, DIR_EDA)

    def test_run(self):
        preprocessor = Preprocessor(COLS_NUM, ["merchantZip"], 
                                    PATH_READ_X, PATH_READ_y, DIR_EDA)
        preprocessor.run(hot_encode=False, normalize=False)

    def test_label_encoding(self):
        preprocessor = Preprocessor(COLS_NUM, ["merchantZip"], 
                                    PATH_READ_X, PATH_READ_y, DIR_EDA)
        preprocessor.run(label_encode=True)

    def test_one_hot_encoding(self):
        preprocessor = Preprocessor(COLS_NUM, ["merchantZip"], 
                                    PATH_READ_X, PATH_READ_y, DIR_EDA)
        preprocessor.run(hot_encode=True)

    def test_normalize(self):
        preprocessor = Preprocessor(COLS_NUM, ["merchantZip"], 
                                    PATH_READ_X, PATH_READ_y, DIR_EDA)
        preprocessor.run(normalize=True)

  
