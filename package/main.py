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

from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_data(target: str="rating"):
    # get the data
    if target == "rating":
        data_X = pd.read_csv("/home/clement/code/Fholklo/GameForecast/raw_data/X_train.csv") #/home/clement/code/Fholklo/GameForecast/raw_data/X_train.csv
        data_Y = pd.read_csv("/home/clement/code/Fholklo/GameForecast/raw_data/y_train.csv")
        y = data_Y["Rating"].copy()
        print("✅ Get train dataset for rating target \n")

    elif target == "player":
        data_X = pd.read_csv("/home/clement/code/Fholklo/GameForecast/raw_data/X_train_player.csv")
        data_Y = pd.read_csv("/home/clement/code/Fholklo/GameForecast/raw_data/y_train_player.csv")
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
        data_X = pd.read_csv("/home/clement/code/Fholklo/GameForecast/raw_data/X_test.csv")
        data_Y = pd.read_csv("/home/clement/code/Fholklo/GameForecast/raw_data/y_test.csv")
        y = data_Y["Rating"].copy()
        y = tf.convert_to_tensor(y.to_numpy(),dtype='float')

        print("✅ Get test dataset for rating target \n")

    elif target == "player":
        data_X = pd.read_csv("/home/clement/code/Fholklo/GameForecast/raw_data/X_test_player.csv")
        data_Y = pd.read_csv("/home/clement/code/Fholklo/GameForecast/raw_data/y_test_player.csv")
        y = data_Y["Peak Players"].copy()
        y = np.log(1 + y)
        y = tf.convert_to_tensor(y.to_numpy(),dtype='float')
        print("✅ Get test dataset for player target \n")

    else:
        print(f"\n❌ Incorrect target name, choose 'rating' or 'player'")
        return None
    return data_X, y

def preprocess(X: pd.DataFrame) -> np.ndarray :
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Process data
    X_clean = clean_data(X,train=True)

    preprocessor = full_preprocessor()

    X_preprocess = preprocessor.fit_transform(X_clean)

    #token pour le texte
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_clean["About_The_Game"])
    sequences = tokenizer.texts_to_sequences(X["About_The_Game"])
    text_tokenize = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='post', truncating='post')


    numeric_input = X_preprocess.drop(columns = ["remainder__About_The_Game","remainder__Screenshots"]).to_numpy()
    text_input = text_tokenize
    image_input = X_preprocess["remainder__Screenshots"].to_numpy()

    print("✅ preprocess() done \n")

    return preprocessor, tokenizer, numeric_input,text_input,image_input

def preprocess_test(preprocessor,tokenizer, X: pd.DataFrame) -> tf.Tensor :
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Process data
    X_clean = clean_data(X,train=True)

    X_preprocess = preprocessor.transform(X_clean)

    sequences = tokenizer.texts_to_sequences(X["About_The_Game"])
    text_tokenize = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='post', truncating='post')

    numeric_input = X_preprocess.drop(columns = ["remainder__About_The_Game","remainder__Screenshots"]).to_numpy()
    text_input = text_tokenize
    image_input = X_preprocess["remainder__Screenshots"].to_numpy()

    print("✅ preprocess_test() done \n")

    return numeric_input,text_input, image_input

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def prepare_for_training(numerical, text, image, y):
    return {'numerical_input': numerical, 'text_input': text, 'img_input': image}, y

def train(numeric_input: np.ndarray,
          text_input: np.ndarray,
          image_input: np.ndarray,
        y_train: pd.DataFrame,
        target: str="rating",
        learning_rate=0.0001,
        batch_size = 32,
        patience = 20,
        validation_split = 0.2
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """
    y_train = y_train.to_numpy()

    # Splitting the data for training and validation
    numeric_train_np, numeric_val_np, text_train_np, text_val_np, image_train_np, \
        image_val_np, y_train_np, y_val_np = train_test_split(
        numeric_input, text_input, image_input, y_train, test_size=validation_split)

    #Numpy to tensor
    numeric_train = tf.convert_to_tensor(numeric_train_np, dtype='float')
    numeric_val = tf.convert_to_tensor(numeric_val_np, dtype='float')
    text_train = tf.convert_to_tensor(text_train_np, dtype='list')
    text_val = tf.convert_to_tensor(text_val_np, dtype='list')
    image_train = tf.convert_to_tensor(image_train_np, dtype='string')
    image_val = tf.convert_to_tensor(image_val_np, dtype='string')
    y_train_tf = tf.convert_to_tensor(y_train_np, dtype='float')
    y_val_tf = tf.convert_to_tensor(y_val_np, dtype='float')

    # Creating TensorFlow datasets
    train_data = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(numeric_train),
        tf.data.Dataset.from_tensor_slices(text_train),
        tf.data.Dataset.from_tensor_slices(image_train).map(load_and_preprocess_image),
        tf.data.Dataset.from_tensor_slices(y_train_tf)
    ))
    val_data = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(numeric_val),
        tf.data.Dataset.from_tensor_slices(text_val),
        tf.data.Dataset.from_tensor_slices(image_val).map(load_and_preprocess_image),
        tf.data.Dataset.from_tensor_slices(y_val_tf)
    ))
    train_dataset = train_data.map(prepare_for_training()).batch(batch_size)
    val_dataset = val_data.map(prepare_for_training).batch(batch_size)

    # Train model using `model.py`
    model = initialize_model_v2(input_shape_num=numeric_train.shape[-1], MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                                input_shape_img=(256,256,3))

    compiled_model = compile_model(model,target=target,learning_rate=learning_rate)

    trained_model, history = train_model(
        compiled_model, train_dataset,
        batch_size=batch_size,
        patience=patience,
        validation_data=val_dataset
    )

    val_mse = np.min(history.history['val_loss'])
    if target == "rating":
        val_mae = np.min(history.history['val_mae'])
    if target == "player":
        val_mae = np.min(history.history['val_loss'])

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model_name=f"model_{target}",model=trained_model)

    print("✅ train() done \n")

    return trained_model, val_mae, val_mse

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

    preprocessor, tokeniser, numeric_input,text_input,image_input = preprocess(data_X[:200])

    trained_model, _, _ = train(numeric_input=numeric_input,
          text_input=text_input,
          image_input=image_input,
          y_train=data_Y[:200],
          target="rating",
          learning_rate=0.0001,
          batch_size = 32,
          patience = 10,
          validation_split = 0.2)

    # testing rating model
    #data_X_test, y_test = get_test_data(target="rating")

    #X_test = preprocess_test(preprocessor=preprocessor,X=data_X_test)

    #evaluate_model(trained_model,X_test,y_test,batch_size=32)


    # training player model
    # data_X, data_Y = get_data(target="player")

    # preprocessor, numeric_input,text_input,image_input = preprocess(data_X)

    # trained_model, _, _ = train(numeric_input=numeric_input,
    #       text_input=text_input,
    #       image_input=image_input,
    #       y_train=data_Y,
    #       target="player",
    #       learning_rate=0.0001,
    #       batch_size = 32,
    #       patience = 20,
    #       validation_split = 0.2)

    # testing rating model
    #data_X_test, y_test = get_test_data(target="player")

    #X_test = preprocess_test(preprocessor=preprocessor,X=data_X_test)

    #evaluate_model(trained_model,X_test,y_test,target="player",batch_size=32)

    #pred()
