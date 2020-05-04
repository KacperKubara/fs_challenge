""" Script to run a whole pipeline 
(EDA + Preprocessing + ML)"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, \
                            precision_score, recall_score

from data_preprocessing import Preprocessor
from eda import EDA
from config import *

pipeline_dict = {
    "rf":
    {
        "model": RandomForestClassifier(),
        "model_params": {
            "n_estimators": [100, 250],
            "max_depth": [5, 10, 15]
        },
        "preprocessing_params": {
            "label_encode": True,
            "hot_encode": False,
            "normalize": True,
            "save": False
        }
    },
}


if __name__ == "__main__":
    eda_runner = EDA(COLS_NUM,
                     PATH_READ_X,
                     PATH_READ_y,
                     DIR_EDA)
    preprocessor = Preprocessor(COLS_NUM, 
                                ["merchantZip"],
                                PATH_READ_X,
                                PATH_READ_y,
                                DIR_EDA)
    eda_runner.run()
    
    # Log results
    results = {
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "model": []
    }
    # Iterate over different models
    for model_name, model_dict in pipeline_dict.items():
        data = preprocessor.run(**model_dict["preprocessing_params"])
        
        # Define X and y
        y = data["fraud"]
        X = data.drop(columns= COLS_OTHER + ["fraud"])
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4)
        
        # Train and predict
        model = model_dict["model"]
        clf = GridSearchCV(model, model_dict["model_params"])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Get scores
        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["f1"].append(f1_score(y_test, y_pred))
        results["precision"].append(precision_score(y_test, y_pred))
        results["recall"].append(recall_score(y_test, y_pred))
        results["model"].append(model_name)
        print(results)

    # Save scores
    pd.DataFrame(results).to_csv("results.csv")
