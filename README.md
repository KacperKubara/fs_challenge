# fs_challenge
## Installation
1) Run 'conda env create -f environment.yml' to install all packages used in the project. If that doesnt work, install them manually from 'environmen.yml' file

2) Update parameters and paths in the 'config.py' file 

3) To run tests type 'python -m unittest discover'

4) To re-run the model training, type 'python main.py' (it might take a while though)

## Quick description of the repo
*main.py*: script to train SVM and RandomForest Classifier with a hyperparameter optimization. Produces accuracy, f1, precision, and recall scores

*eda.py*: Creates basic visualization and stats from datasets that are saved in 'eda_results' folder

*data_preprocessing*: Runs basic preprocessing, i.e. creates labels, impute missing values, encodes and normalizes data

*config.py*: script with config parameters

*data*: folder with data for the challenge
