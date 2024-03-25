import numpy as np
import time
import tensorflow as tf

from colorama import Fore, Style
from typing import Tuple
from tensorflow import keras
from keras import models,layers, regularizers, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Concatenate, Flatten, LSTM, Rescaling, Reshape, Dropout
from keras.applications import VGG16
from keras.preprocessing.sequence import pad_sequences

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def initialize_model(input_shape: tuple) :
    """
    Initialize the Neural Network with random weights
    """
    #reg = regularizers.l1_l2(l2=0.005)

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(128, activation="relu")) #, kernel_regularizer=reg ?
    #model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(1, activation="linear"))

    print("✅ Model initialized")

    return model

def initialize_model_v2(input_shape_num: int, MAX_SEQUENCE_LENGTH: int, input_shape_img: tuple) :
    """
    Initialize the Neural Network for 3 data types
    """
    numerical_input = Input(shape=input_shape_num, name='numerical_input')
    text_input = Input(shape=MAX_SEQUENCE_LENGTH, name='text_input')
    img_input = Input(shape=input_shape_img, name='img_input')
    rescale = Rescaling(scale=1/255)(img_input)
    reshape = Reshape((255,255,3))(rescale)

    # Text embedding layer
    embedding_dim = 100  # Example dimension for embeddings
    vocab_size = 10000   # Example vocabulary size
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    lstm_layer = LSTM(10, return_sequences=True)(embedding_layer)
    lstm_layer_2 = LSTM(10)(lstm_layer)
    flattened_text = Flatten()(lstm_layer_2)
    # Feature extraction layer for numerical data
    numerical_features = Dense(128, activation='relu')(numerical_input)
    numerical_features = Dense(64, activation='relu')(numerical_features)
    numerical_features = Dense(32, activation='relu')(numerical_features)
    numerical_features = Dense(16, activation='relu')(numerical_features)
    numerical_features = Dropout(rate=0.2)
    # Images
    img_features = VGG16(weights="imagenet", include_top=False)(reshape)
    img_features.trainable = False
    img_features = layers.Flatten()(img_features)
    img_features = layers.BatchNormalization(momentum=0.8)(img_features)
    img_features = layers.Dense(500, activation='relu')(img_features)
    img_features = layers.Dropout(0.5)(img_features)
    # Concatenate numerical, text and image features
    concatenated_features = Concatenate()([numerical_features, flattened_text, img_features])
    # Additional intermediate layers
    hidden_layer = Dense(256, activation='relu')(concatenated_features)
    hidden_layer = Dropout(rate=0.3)
    hidden_layer = Dense(128, activation='relu')(concatenated_features)
    hidden_layer = Dropout(rate=0.3)
    hidden_layer = Dense(64, activation='relu')(concatenated_features)
    hidden_layer = Dropout(rate=0.3)
    hidden_layer = Dense(32, activation='relu')(concatenated_features)
    hidden_layer = Dropout(rate=0.3)
    output_layer = Dense(1, activation='linear')(hidden_layer)
    # Define the model
    model = Model(inputs=[numerical_input, text_input, img_input], outputs=output_layer)
    # ce bloc est notre modèle
    print(":coche_blanche: Model initialized")
    return model


def compile_model(model, target:str=None, learning_rate=0.0001) :
    """
    Compile the Neural Network
    """

    initial_learning_rate = 0.0001
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1600,
        decay_rate=0.7,
        staircase=True)

    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    #optimizer = optimizers.Adam(learning_rate=learning_rate)

    if target == "rating:":
        model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
    else:
        model.compile(loss="mean_squared_logarithmic_error", optimizer=optimizer, metrics=["mean_squared_logarithmic_error","mae"])

    print("✅ Model compiled")

    return model

def train_model(
        model,
        X,
        y,
        target:str="rating",
        batch_size=32,
        patience=10,
        validation_split=0.2,
    ) :
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    rp = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=5, min_lr=0.000005)

    history = model.fit(
        X,
        y,
        validation_split=validation_split,
        epochs=150,
        batch_size=batch_size,
        callbacks=[es,rp],
        verbose=1
    )

    if target == "rating":
        print(f"""✅ Model rating trained with : min val MAE: {round(np.min(history.history['val_mae']), 2)} \n

                                    : loss: {round(np.min(history.history['val_loss']), 2)}""")

    if target == "player":
        print(f"✅ Model player trained with : min val RMSLE: {round(np.min(history.history['val_loss']), 2)}")

    return model, history
