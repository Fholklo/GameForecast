import numpy as np
import time
import tensorflow as tf

from colorama import Fore, Style
from typing import Tuple
from tensorflow import keras
from keras import models,layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

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


def compile_model(model, target:str=None, learning_rate=0.0001) :
    """
    Compile the Neural Network
    """

    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1600,
        decay_rate=0.7,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
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
        batch_size=128,
        patience=20,
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

    rp = tf.keras.callbacks.ReduceLROnPlateau(
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
