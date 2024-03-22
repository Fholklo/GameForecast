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

def get_data(target: str="rating"):
    # get the data
    if target == "rating":
        data_X = pd.read_csv("/home/nicolas/code/NicolasAdrs/projet/GameForecast/raw_data/X_train.csv") #/home/clement/code/Fholklo/GameForecast/raw_data/X_train.csv
        data_Y = pd.read_csv("/home/nicolas/code/NicolasAdrs/projet/GameForecast/raw_data/y_train.csv")
        y = data_Y["Rating"].copy()
        print("✅ Get train dataset for rating target \n")

    elif target == "player":
        data_X = pd.read_csv("/home/nicolas/code/NicolasAdrs/projet/GameForecast/raw_data/X_train_player.csv")
        data_Y = pd.read_csv("/home/nicolas/code/NicolasAdrs/projet/GameForecast/raw_data/y_train_player.csv")
        y = data_Y["Peak Players"].copy()
        y = np.log(1 + y)
        print("✅ Get train dataset for player target \n")

    else:
        print(f"\n❌ Incorrect target name, choose 'rating' or 'players'")
        return None
    return data_X, y

def get_test_data(target: str="rating"):
    # get the test data
    if target == "rating":
        data_X = pd.read_csv("/home/nicolas/code/NicolasAdrs/projet/GameForecast/raw_data/X_test.csv")
        data_Y = pd.read_csv("/home/nicolas/code/NicolasAdrs/projet/GameForecast/raw_data/y_test.csv")
        y = data_Y["Rating"].copy()
        y = tf.convert_to_tensor(y.to_numpy(),dtype='float')

        print("✅ Get test dataset for rating target \n")

    elif target == "player":
        data_X = pd.read_csv("/home/nicolas/code/NicolasAdrs/projet/GameForecast/raw_data/X_test_player.csv")
        data_Y = pd.read_csv("/home/nicolas/code/NicolasAdrs/projet/GameForecast/raw_data/y_test_player.csv")
        y = data_Y["Peak Players"].copy()
        y = np.log(1 + y)
        y = tf.convert_to_tensor(y.to_numpy(),dtype='float')
        print("✅ Get test dataset for player target \n")

    else:
        print(f"\n❌ Incorrect target name, choose 'rating' or 'player'")
        return None
    return data_X, y

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

    return preprocessor, X_preprocess

def preprocess_test(preprocessor, X: pd.DataFrame) -> pd.DataFrame :
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

    X_preprocess = preprocessor.transform(X_clean)

    X_preprocess = tf.convert_to_tensor(X_preprocess.to_numpy(),dtype='float')

    print("✅ preprocess_test() done \n")

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
        target: str="rating",
        learning_rate=0.0001,
        batch_size = 128,
        patience = 20,
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
    model = initialize_model(X_train_tf.shape[-1])

    compiled_model = compile_model(model,target=target,learning_rate=learning_rate)

    trained_model, history = train_model(
        compiled_model, X_train_tf, y_train_tf,
        batch_size=batch_size,
        patience=patience,
        validation_split=validation_split
    )

    val_mse = np.min(history.history['val_loss'])
    if target == "rating":
        val_mae = np.min(history.history['val_mae'])
    if target == "player":
        val_mae = np.min(history.history['val_loss'])

    #params = dict(context="train",training_set_size=DATA_SIZE,row_count=len(X_train_processed))

    # Save results on the hard drive using taxifare.ml_logic.registry
    # save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model_name=f"model_{target}",model=trained_model)

    print("✅ train() done \n")

    return trained_model, X_train_tf, y_train_tf, val_mae, val_mse, history

def evaluate_model(
        model,
        X,
        y,
        target: str="rating",
        batch_size=32
    ) :
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    if target == "rating":
        mae = metrics["mae"]
        print(f"✅ Model rating evaluated, MAE: {round(mae, 2)}, MSE loss {round(loss, 2)}")

    if target == "player":
        print(f"✅ Model rating evaluated, RMSE loss {round(loss, 2)}")

    return metrics


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

    # training rating model
    data_X, data_Y = get_data(target="rating")

    preprocessor, X_preprocess = preprocess(data_X)

    trained_model, X_train_tf, y_train_tf, _, _, history = train(X_preprocess,
          data_Y,
          target="rating",
          learning_rate=0.0001,
          batch_size = 128,
          patience = 10,
          validation_split = 0.2)

    # testing rating model
    data_X_test, y_test = get_test_data(target="rating")

    X_test = preprocess_test(preprocessor=preprocessor,X=data_X_test)

    evaluate_model(trained_model,X_test,y_test,batch_size=32)


    # training player model
    data_X, data_Y = get_data(target="player")

    preprocessor, X_preprocess = preprocess(data_X)

    trained_model, X_train_tf, y_train_tf, _, _, history = train(X_preprocess,
          data_Y,
          target="player",
          learning_rate=0.0001,
          batch_size = 128,
          patience = 20,
          validation_split = 0.2)

    # testing rating model
    data_X_test, y_test = get_test_data(target="player")

    X_test = preprocess_test(preprocessor=preprocessor,X=data_X_test)

    evaluate_model(trained_model,X_test,y_test,target="player",batch_size=32)

    #pred()
