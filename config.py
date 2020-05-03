""" Sets up all dir paths and global parameters"""
PATH_READ_X = "./data/transactions_obf.csv"
PATH_READ_y = "./data/labels_obf.csv"
DIR_EDA = "./eda_results"

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
