import numpy as np
import pandas as pd

import os
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from package.scripts.params import *
from package.scripts.preprocessor import *
from package.ml_logics.model import *
from package.ml_logics.registry import *

import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_data():
    # get the data
    data_X = pd.read_csv("/home/clement/code/Fholklo/GameForecast/raw_data/X_train.csv")
    data_Y = pd.read_csv("/home/clement/code/Fholklo/GameForecast/raw_data/y_train.csv")
    y_rating = data_Y["Rating"].copy()
    return data_X, y_rating

def preprocess(X: pd.DataFrame) -> pd.DataFrame :
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Process data
    X_clean = clean_data(X)

    preprocessor = full_preprocessor()

    X_preprocess = preprocessor.fit_transform(X_clean)

    print("✅ preprocess() done \n")

    return X_preprocess

# def consistency_XY(App_ID: pd.DataFrame, data_Y: pd.DataFrame) -> pd.DataFrame:
#     '''consistent features - target'''
#     #data_X = data_X[data_X['App_ID'].isin(y['App_ID'])]
#     y_rating = App_ID.copy()
#     y = data_Y[data_Y['App_ID'].isin(y_rating['App_ID'])]

#     y_rating.drop(columns='App_ID',inplace=True)
#     y.drop(columns='App_ID',inplace=True)

#     return y_rating, y

def train(X_train_preprocess: pd.DataFrame,
          y_train: pd.DataFrame,
        learning_rate=0.0005,
        batch_size = 128,
        patience = 10,
        validation_split = 0.2
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    # Split training and testing data
    # X_train, X_test, y_train, y_test = train_test_split(X_preprocess, y, test_size=0.30)

    X_train_tf = tf.convert_to_tensor(X_train_preprocess.to_numpy(),dtype='float')
    y_train_tf = tf.convert_to_tensor(y_train.to_numpy(),dtype='float')

    # Train model using `model.py`
    model = load_model(model_name='model_rating')

    if model is None:
        model = initialize_model(X_train_tf.shape[-1])

    compiled_model = compile_model(model,learning_rate=learning_rate)

    trained_model, history = train_model(
        compiled_model, X_train_tf, y_train_tf,
        batch_size=batch_size,
        patience=patience,
        validation_split=validation_split
    )

    val_mae = np.min(history.history['val_mae'])
    val_mse = np.min(history.history['val_loss'])

    #params = dict(context="train",training_set_size=DATA_SIZE,row_count=len(X_train_processed))

    # Save results on the hard drive using taxifare.ml_logic.registry
    # save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model_name="model_rating",model=trained_model)

    print("✅ train() done \n")

    return val_mae, val_mse

"""def evaluate() -> float:

    data_processed = data_processed.to_numpy()

    X_new = data_processed[:, :-1]
    y_new = data_processed[:, -1]

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae
"""

"""
def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
        pickup_longitude=[-73.950655],
        pickup_latitude=[40.783282],
        dropoff_longitude=[-73.984365],
        dropoff_latitude=[40.769802],
        passenger_count=[1],
    ))

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred
"""

if __name__ == '__main__':
    data_X, data_Y = get_data()

    X_preprocess = preprocess(data_X)

    train(X_preprocess,
          data_Y,
          learning_rate=0.0005,
          batch_size = 128,
          patience = 10,
          validation_split = 0.2)
    #evaluate()
    #pred()
