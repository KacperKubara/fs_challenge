import unittest

from eda import EDA
from config import COLS_NUM, PATH_READ_X, PATH_READ_y, DIR_EDA


class TestEDA(unittest.TestCase):
    """ Class for testing EDA class"""
    def test_compile(self):
        eda_runner = EDA(COLS_NUM, PATH_READ_X, PATH_READ_y, DIR_EDA)

    def test_run(self):
        eda_runner = EDA(COLS_NUM, PATH_READ_X, PATH_READ_y, DIR_EDA)
        eda_runner.run()
