import numpy as np
import time

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
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(64, activation="relu"))
    #model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(1, activation="linear"))

    print("✅ Model initialized")

    return model


def compile_model(model, learning_rate=0.0005) :
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="neg_mean_squared_error", optimizer=optimizer, metrics=["mae","accuracy"])

    print("✅ Model compiled")

    return model

def train_model(
        model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=32,
        patience=5,
        validation_data=None,
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

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


def evaluate_model(
        model,
        X: np.ndarray,
        y: np.ndarray,
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
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
