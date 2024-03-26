import numpy as np
import time
import tensorflow as tf

from colorama import Fore, Style
from keras import models,layers, regularizers, optimizers
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, LSTM, Rescaling, Reshape, Dropout, Input, Concatenate
from keras.applications import VGG16
from keras.preprocessing.sequence import pad_sequences

def initialize_model_numeric(input_shape: int) :
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

    print("✅ Model initialized_numeric")

    return model

def initialize_model_text(input_dim: int,max_len:int) :
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=64, input_length=max_len),
        LSTM(64),
        Dense(1, activation='linear')
    ])
    print("✅ Model initialized")

    return model

def set_nontrainable_layers(model):
    model.trainable = True
    return model
def load_vgg_model(input_shape=(128, 128, 3)):
    model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    model = set_nontrainable_layers(model)
    return model
def add_last_layers(model):
    '''Take a pre-trained model'''
    flatten_layer = layers.Flatten()
    batch_norm_layer = layers.BatchNormalization()
    dense_layer = layers.Dense(500, activation='relu')
    drop_out_layer = layers.Dropout(0.3)
    dense_layer = layers.Dense(128, activation='relu')
    drop_out_layer = layers.Dropout(0.3)
    dense_layer = layers.Dense(64, activation='relu')
    drop_out_layer = layers.Dropout(0.3)
    prediction_layer = layers.Dense(1, activation='linear')
    model = models.Sequential([
        model,
        flatten_layer,
        batch_norm_layer,
        dense_layer,
        drop_out_layer,
        prediction_layer
    ])
    return model
def initialize_cnn_model():
    model = load_vgg_model()
    model = add_last_layers(model)
    return model


def initialize_metamodel(input_shape_base_pred) :
    """
    Initialize the Neural Network with random weights
    """
      # Remplacez par la forme réelle des prédictions
    base_pred_input1 = Input(shape=(1,), name='base_pred_input1')
    base_pred_input2 = Input(shape=(1,), name='base_pred_input2')
    base_pred_input3 = Input(shape=(1,), name='base_pred_input3')

    # Concaténez les prédictions pour former les caractéristiques du métamodèle
    concatenated_features = Concatenate()([base_pred_input1, base_pred_input2,base_pred_input3])

    # Définissez le métamodèle
    meta_hidden_layer = Dense(128, activation='relu')(concatenated_features)
    meta_hidden_layer = Dense(64, activation='relu')(meta_hidden_layer)
    meta_hidden_layer = Dense(32, activation='relu')(meta_hidden_layer)
    meta_hidden_layer = Dropout(rate=0.3)(meta_hidden_layer)
    meta_hidden_layer = Dense(8, activation='relu')(meta_hidden_layer)
    meta_output_layer = Dense(1, activation='linear')(meta_hidden_layer)  # Ajustez selon votre tâche

    # Créez le modèle de stacking
    meta_model = Model(inputs=[base_pred_input1, base_pred_input2,base_pred_input3], outputs=meta_output_layer)


    return meta_model


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

    if target == "rating":
        model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
    else:
        model.compile(loss="mean_squared_logarithmic_error", optimizer=optimizer, metrics=["mean_squared_logarithmic_error","mae"])

    print("✅ Model compiled")

    return model

def train_model_numeric(
        model,
        X_num,
        Y_num,
        target:str="rating",
        batch_size=32,
        patience=10,
        validation_split=0.2
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
        x=X_num,
        y=Y_num,
        validation_split=validation_split,
        epochs=150,
        batch_size=batch_size,
        callbacks=[es,rp],
        verbose=1
    )

    if target == "rating":
        print(f"""✅ Model_num rating trained with : min val MAE: {round(np.min(history.history['val_mae']), 2)} \n

                                    : loss: {round(np.min(history.history['val_loss']), 2)}""")

    if target == "player":
        print(f"✅ Model_num player trained with : min val RMSLE: {round(np.min(history.history['val_loss']), 2)}")

    return model, history

def train_model_text(
        model,
        X_text,
        Y_text,
        target:str="rating",
        batch_size=32,
        patience=10,
        validation_split=0.2
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
        x=X_text,
        y=Y_text,
        validation_split=validation_split,
        epochs=150,
        batch_size=batch_size,
        callbacks=[es,rp],
        verbose=1
    )

    if target == "rating":
        print(f"""✅ Model_text rating trained with : min val MAE: {round(np.min(history.history['val_mae']), 2)} \n

                                    : loss: {round(np.min(history.history['val_loss']), 2)}""")

    if target == "player":
        print(f"✅ Model_text player trained with : min val RMSLE: {round(np.min(history.history['val_loss']), 2)}")

    return model, history

def train_model_image(
        model,
        X_image,
        Y_image,
        target:str="rating",
        batch_size=32,
        patience=10,
        validation_split=0.2
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
        x=X_image,
        y=Y_image,
        validation_split=validation_split,
        epochs=150,
        batch_size=batch_size,
        callbacks=[es,rp],
        verbose=1
    )

    if target == "rating":
        print(f"""✅ Model_text rating trained with : min val MAE: {round(np.min(history.history['val_mae']), 2)} \n

                                    : loss: {round(np.min(history.history['val_loss']), 2)}""")

    if target == "player":
        print(f"✅ Model_text player trained with : min val RMSLE: {round(np.min(history.history['val_loss']), 2)}")

    return model, history




def train_metamodel(
        model,
        X_meta ,
        Y_meta,
        target:str="rating",
        batch_size=32,
        patience=10,
        validation_split=0.2
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
        x=X_meta,
        y=Y_meta,
        validation_split=validation_split,
        epochs=150,
        batch_size=batch_size,
        callbacks=[es,rp],
        verbose=1
    )

    if target == "rating":
        print(f"""✅ Model_num rating trained with : min val MAE: {round(np.min(history.history['val_mae']), 2)} \n

                                    : loss: {round(np.min(history.history['val_loss']), 2)}""")

    if target == "player":
        print(f"✅ Model_num player trained with : min val RMSLE: {round(np.min(history.history['val_loss']), 2)}")

    return model, history
