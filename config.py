""" Sets up all dir paths and global parameters"""
# Path to transactions_obf.csv
PATH_READ_X = "./data/transactions_obf.csv"
# Path to labels_obf.csv
PATH_READ_y = "./data/labels_obf.csv"
# Folder path for EDA
DIR_EDA = "./eda_results"

# Column names from data
COLS_NUM = [
    "transactionAmount", 
    "availableCash"
    ]
COLS_CAT  = [
    "mcc", 
    "posEntryMode", 
    "merchantCountry", 
    "merchantZip"
    ]
COLS_OTHER = [
    "transactionTime", 
    "eventId", 
    "accountNumber", 
    "merchantId"
    ]
