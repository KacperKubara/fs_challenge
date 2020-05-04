from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pandas as pd
import numpy as np

from runner import Runner


class EDA(Runner):
    def __init__(self, cols_num: List, path_read_X: str, 
                 path_read_y: str, dir_write: str):
        """ Exploratory Data Analysis class
        
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
        self.X_fraud = self.X[self.X["eventId"].isin(
                              self.y["eventId"].unique())]
        self.X_not_fraud = self.X[~ self.X["eventId"].isin(
                                  self.y["eventId"].unique())]
        self.dir_write = dir_write

    def run(self) -> None:
        """ Runs all EDA utilites"""
        self.get_basic_stats()
        self.visualize_missingness()
        self.payment_distribution()


    def get_basic_stats(self) -> None:
        """ Saves basics data stats using pd.describe() with 
        whole data, fraudulent data, and not fraudulent data"""
        # Whole data
        # Get stats for numerical X cols
        self.X[self.cols_num].astype('float').\
            describe().to_csv(self.dir_write + "/X_stats_num.csv")
        # Get stats for cat X cols
        self.X.drop(self.cols_num, axis=1).astype('object').\
            describe().to_csv(self.dir_write + "/X_stats_cat.csv")
        # Get stats for y
        self.y.describe().to_csv(self.dir_write + "\y_stats.csv")

        # Fraudulent data
        self.X_fraud[self.cols_num].astype('float').\
            describe().to_csv(self.dir_write + "/X_stats_num_fraud.csv")
        # Get stats for cat X cols
        self.X_fraud.drop(self.cols_num, axis=1).astype('object').\
            describe().to_csv(self.dir_write + "/X_stats_cat_fraud.csv")

        # Not fraudulent data
        self.X_not_fraud[self.cols_num].astype('float').\
            describe().to_csv(self.dir_write + "/X_stats_num_not_fraud.csv")
        # Get stats for cat X cols
        self.X_not_fraud.drop(self.cols_num, axis=1).astype('object').\
            describe().to_csv(self.dir_write + "/X_stats_cat_not_fraud.csv")

    def visualize_missingness(self) -> None:
        """ Visualize missingness distribution in data"""
        # It looks only for NaN and 0s, usually missigness can be represented 
        # by other characters as well that are column-specific, 
        # but I don't have much time to analyze data thorougly
        zip_code_regex = "[A-Za-z]{1,}[0-9]{1,}" # Valid zipcode expressed by regex
        X_converted = self.X
        # Convert all elements that dont match zipcode regex to NaNs
        # Temporary conversion from Nan to False to use Series as a bit mask
        X_mask = \
            ~X_converted["merchantZip"].str.match(zip_code_regex).fillna(False)
        X_converted.loc[X_mask, "merchantZip"] = np.nan
        # Visualize missingness distribution
        ax = msno.matrix(X_converted)
        fig = plt.gcf()
        fig.savefig(self.dir_write + "/missingness.png")
        plt.clf()

    def payment_distribution(self) -> None:
        """ Visualize payment distribution for data with and without fraud"""
        std_trans = self.X["transactionAmount"].std()
        mean_trans = self.X["transactionAmount"].mean()
        # Whole data
        fig, ax = plt.subplots(1,2)
        X_no_outliers = self.X[self.X["transactionAmount"] 
                               < mean_trans + 3*std_trans]
        sns.distplot(X_no_outliers["transactionAmount"], ax=ax[0])
        sns.distplot(self.X["availableCash"], ax=ax[1])
        fig.savefig(self.dir_write + "/payment_dist.png")
        plt.clf()

        # Fraudulent data
        fig, ax = plt.subplots(1,2)
        X_no_outliers = self.X_fraud[self.X_fraud["transactionAmount"] 
                                     < mean_trans + 3*std_trans]
        sns.distplot(X_no_outliers["transactionAmount"], ax=ax[0])
        sns.distplot(self.X_fraud["availableCash"], ax=ax[1])
        fig.savefig(self.dir_write + "/payment_dist_fraud.png")
        plt.clf()

        # Non fraudulent data
        fig, ax = plt.subplots(1,2)
        X_no_outliers = self.X_not_fraud[self.X_not_fraud["transactionAmount"]
                                         < mean_trans + 3*std_trans]
        sns.distplot(X_no_outliers["transactionAmount"], ax=ax[0])
        sns.distplot(self.X_not_fraud["availableCash"], ax=ax[1])
        fig.savefig(self.dir_write + "/payment_dist_not_fraud.png")
        plt.clf()
