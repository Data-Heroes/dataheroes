import time
from itertools import product
from pathlib import Path
from typing import Union
import os 
import random
import tqdm

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss
import pandas as pd
from pandas.core.series import Series
import numpy as np

from dataheroes import CoresetTreeServiceDTC

from pyspark.sql import SparkSession
from pyspark.broadcast import Broadcast
import gdown

"""
This code is an example of how to use the CoresetTreeServiceDTC to train a model using the coreset tree and evaluate the model using holdout validation

on spark, where we are training the model using different hyperparameters in parallel. Instead of splitting the dataset across the cluster nodes,

each node in the cluster receives a copy of the Coreset data and trains some of the hyperparameter combinations on it.

In other words, the hyperparameter combinations are split across the cluster and not the data itself.

We are using a small part of the Criteo dataset (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) which is a binary classification dataset.

XGBClassifier is used as the model. For evaluation, we are using auc and log_loss as the metrics.

Note:
This file is to be passed to a spark cluster and be submitted as a pyspark application. 
The cluster should be configured with the necessary dependencies.

Example of how to submit the script to the cluster:
spark-submit --master yarn --deploy-mode cluster gridsearch_parallel_hyperparameters_holdout_validate.py
"""

#Initialize spark session. Change the master to the desired master url. 
spark = SparkSession.builder.appName("gridsearch_parallel_hyperparameters_holdout_validate") \
    .getOrCreate()

#Define  helper functions for the script.
def calculate_metrics(y_true:Union[Series, np.ndarray], y_pred_proba:Union[np.ndarray, Series]) -> dict:
    """
    Calculate the metrics of the model.

    - y_true: the true target values.
    - y_pred_proba: the predicted target values.

    Returns a dictionary containing the metrics.
    """
    metrics = {}
    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    metrics['log_loss'] = log_loss(y_true, y_pred_proba)

    return metrics

def set_categorical_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns that are categorical to type 'category'.

    - df: pd.DataFrame, the dataframe to be converted.

    Returns the dataframe with the columns converted to type 'category'.

    ### Note:
    This is needed because xgb does not handle categorical features automatically. We need to convert them to type 'category' before training the model.
    """
    for col in df.columns:
        if col.startswith('c'):
            df[col] = df[col].astype('category')
        else:
            df[col] = df[col].astype(float)

    return df

def evaluate_hyperparameters_holdout_validation(params:dict, data_broadcast:Broadcast, target:str) -> tuple:
    """
    Trains a model using the given hyperparameters and evaluates it on the validation data.

    This function is meant to be run on a worker node.

    - params: dict, the hyperparameters to be evaluated.
    - data_broadcast: Broadcast, the training and test data.
    - target: str, the target column. 
    
    Returns a tuple containing the hyperparameters and the scores.

    ### Note:
    Target has to be provided because each worker is located on a different machine and we want to make sure that the target column is available on each worker.
    """
    # Extract the data from the broadcast variable.
    coreset, validation = data_broadcast.value

    #coreset is a dictionary containing the coreset data.
    #validation is a pandas dataframe containing the validation data.

    X = coreset['X']
    y = coreset['y']
    w = coreset['w']

    X_val = validation.drop(target, axis=1)
    y_val = validation[target]
    
    # Convert all columns that are categorical to type 'category'.
    X = set_categorical_features(X)
    X_val = set_categorical_features(X_val)

    model = XGBClassifier(**params)
    
    print(f"Performing holdout validation with params: {params}.")
    t = time.time()
    #Train the model. Coresets are weighted sets of instances, so we need to pass the sample_weight parameter to the fit method.
    model.fit(X, y, sample_weight=w)
    train_time = time.time()-t

    #predict the target.
    proba = model.predict_proba(X_val)[:, 1]

    scores = calculate_metrics(y_val, proba)
    scores['train_time'] = train_time

    return (params, scores) #return the hyperparameters and the scores back to the driver.

def main():
    #Download the data from google drive. The data is a zip file which contains 10 csv files.
    #Each file contains 1 million rows of the criteo dataset.
    url = "https://drive.google.com/uc?id=1SxOgh-fINjXddwszc9bp5UqoBMS427X6"
    data_path = Path('/tmp/criteo10M')
    data_path.mkdir(parents=True, exist_ok=True)
    zip_file = 'criteo10M.zip'
    if not os.path.exists(data_path / 'criteo0.csv'):
        gdown.download(url, output = str(data_path / zip_file), quiet=False, use_cookies=False)
        import zipfile
        with zipfile.ZipFile(data_path / zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)


    #The data is split into 10 files. 
    #We will use the first 6 files for training, the next 2 files for validation, and the last 2 files for testing.

    train_no = 6#Number of files to train on.
    validation_no = 2 #Number of files to validate on.
    test_no = 2 #Number of files to test on.
    train_file = [str(data_path / f'criteo{i}.csv') for i in range(train_no)] 
    validation_file = [str(data_path / f'criteo{i}.csv') for i in range(train_no, train_no+validation_no)]
    test_file = [str(data_path / f'criteo{i}.csv') for i in range(train_no+validation_no, train_no+validation_no+test_no)]

    #Load the dataset into memory.
    train = pd.concat([pd.read_csv(f) for f in tqdm(train_file, desc='Loading train data')])
    validation = pd.concat([pd.read_csv(f) for f in tqdm(validation_file, desc='Loading validation data')])
    test = pd.concat([pd.read_csv(f) for f in tqdm(test_file, desc='Loading test data')])

    #Target of the dataset.
    target = 'label'
    data_params = {
        'target': {'name': target},
    }
    #Choose the desired coreset tree configuration.
    n_instances = train_no * 1_000_000
    chunk_size = n_instances // 8 #This makes sure that we create a tree with 8 leaves at the bottom level.
    coreset_size = 0.2

    #Initialize the coreset tree service.
    service = CoresetTreeServiceDTC(optimized_for='training', 
                                    n_instances=n_instances,
                                    chunk_size=chunk_size,
                                    coreset_size=coreset_size,
                                    data_params=data_params)

    #Build the coreset tree on the train data.
    print("Building the coreset tree on the train data.")
    service.build_from_df(train, copy=True) #We pass copy=true because the build is changing the data.

    #Get the coreset tree in order to pass it to the other workers.
    print("Getting the coreset tree.")
    coreset = service.get_coreset(level=2, as_df=True)

    hyperparameters = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'enable_categorical': [True]
    }

    #The function that is training the model with their hyperparameters evaluates using 2 metrics (auc, log_loss). We choose what metric we want to optimize.
    optimize = 'auc'

    #Create all possible combinations of hyperparameters and shuffle them.
    keys, values = zip(*hyperparameters.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    random.shuffle(param_combinations)

    sc = spark.sparkContext
    #Broadcast the coreset and validation data to all the workers.
    print("Broadcasting the coreset and validation data to the workers.") 
    broadcast_data = sc.broadcast((coreset, validation))

    #Create an rdd with the hyperparameters. This splits the hyperparameter combinations among different tasks.
    print("Performing holdout validation with different hyperparameters in parallel.")
    param_combinations_rdd = sc.parallelize(param_combinations, numSlices=4)  # Adjust numSlices as per the number of nodes/executors in the cluster.

    #Train the model using the hyperparameters in parallel. Each job passed to the worker is a hyperparameter combination.
    results = param_combinations_rdd.map(lambda params: evaluate_hyperparameters_holdout_validation(params, broadcast_data, target)).collect()

    #Get the best hyperparameters and the best score. Note that we are optimizing for auc. x[1] is the score dictionary and x[1][optimize] is the auc score.
    best_hyperparams, best_score= max(results, key=lambda x: x[1][optimize])

    print(f"Best hyperparams: {best_hyperparams}.")
    print(f"Best score: {best_score}.")


    #Train on both the training and validation data for the best hyperparameters.
    X_train = pd.concat([train, validation]).drop(target, axis=1)
    y_train = pd.concat([train, validation])[target]

    X_test = test.drop(target, axis=1)
    y_test = test[target]

    #Setting the categorical features to type 'category' in order for XGBClassifier to handle them.
    X_train = set_categorical_features(X_train)
    X_test = set_categorical_features(X_test)

    #Train a model using the best hyperparameters and test on the test data.
    print("Training a model using the best hyperparameters.")
    model = XGBClassifier(**best_hyperparams)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    test_scores = calculate_metrics(y_test, proba)

    print("Scores on the model with the best hyperparameters.")
    for metric, value in test_scores.items():
        print(f"Full dataset train {metric}: {value}.")

if __name__ == '__main__':
    main()
    spark.stop()