import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle

from colorama import Fore, Style

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from package.scripts.preprocessor import clean_data, full_preprocessor
from package.scripts.params import folder_path,MAX_Len

from package.ml_logics.model import initialize_model_numeric,initialize_cnn_model,initialize_model_text,initialize_metamodel
from package.ml_logics.model import compile_model, train_model_numeric, train_model_image,train_model_text,train_metamodel

from package.ml_logics.registry import save_model, load_most_recent_model, save_results

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

def load_and_preprocess_image(path):
    # Charger l'image à partir du disque
    image = tf.io.read_file(path)
    # Décoder l'image en un tenseur et s'assurer qu'elle a 3 canaux de couleur
    image = tf.image.decode_jpeg(image, channels=3)
    # Redimensionner l'image pour qu'elle corresponde à la taille attendue par le modèle, par exemple (224, 224)
    image = tf.image.resize(image, [128, 128])
    # Normaliser les valeurs des pixels de l'image pour qu'elles soient dans l'intervalle [0, 1]
    image /= 255.0
    # Retourner l'image prétraitée
    return image

def preprocess(X: pd.DataFrame) -> np.ndarray :

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Process data
    X_clean = clean_data(X,train=True)

    preprocessor = full_preprocessor()

    X_preprocess = preprocessor.fit_transform(X_clean)

    #token pour le texte
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_clean["About_The_Game"])
    sequences = tokenizer.texts_to_sequences(X_clean["About_The_Game"])

    text_tokenize = pad_sequences(sequences, maxlen=MAX_Len,padding='post', truncating='post')

    numeric_input = X_preprocess.drop(columns = ["remainder__About_The_Game","remainder__Screenshots"]).to_numpy()

    text_input = text_tokenize

    image_input = X_preprocess["remainder__Screenshots"].to_numpy()
    images_input = np.array([load_and_preprocess_image(path).numpy() for path in image_input])

    file_path_preproc = os.path.join(folder_path, 'preprocessor.pkl')
    with open(file_path_preproc, 'wb') as f:
        pickle.dump(preprocessor, f)
        print("✅ preprocessor saved")
    file_path_tok = os.path.join(folder_path, 'tokenizer.pkl')
    with open(file_path_tok, 'wb') as f:
        pickle.dump(tokenizer, f)
        print("✅ tokenizer saved")

    print("✅ preprocess() done \n")

    return MAX_Len, numeric_input,text_input,images_input

def preprocess_test(X: pd.DataFrame) -> tf.Tensor :

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    #import preproc et tokenizer
    file_path_preproc = os.path.join(folder_path, 'preprocessor.pkl')
    with open(file_path_preproc, 'rb') as f:
        preprocessor = pickle.load(f)

    file_path_tok = os.path.join(folder_path, 'tokenizer.pkl')
    with open(file_path_tok, 'rb') as f:
        tokenizer = pickle.load(f)

    # Process data
    X_clean = clean_data(X,train=True)

    X_preprocess = preprocessor.transform(X_clean)

    #token pour le texte
    sequences = tokenizer.texts_to_sequences(X_clean["About_The_Game"])

    text_tokenize = pad_sequences(sequences, maxlen=MAX_Len,padding='post', truncating='post')

    numeric_input = X_preprocess.drop(columns = ["remainder__About_The_Game","remainder__Screenshots"]).to_numpy()
    text_input = text_tokenize
    image_input = X_preprocess["remainder__Screenshots"].to_numpy()
    images_input = np.array([load_and_preprocess_image(path).numpy() for path in image_input])

    print("✅ preprocess_test() done \n")

    return numeric_input,text_input, images_input

def train_numeric(
        numeric_input: np.ndarray,
        text_input,
        image_input,
        y_train: pd.DataFrame,
        target: str="rating",
        batch_size = 32,
        patience = 20,
        validation_split = 0.2
    ) -> float:

    y_train = y_train.to_numpy()

    numeric_input_train, numeric_input_val,text_input_train, text_input_val,image_input_train,image_input_val,\
        y_train_train,y_train_val = train_test_split(numeric_input,text_input,image_input,y_train, test_size=0.3)

    model = initialize_model_numeric(input_shape=numeric_input_train.shape[-1])

    compiled_model = compile_model(model,target=target)

    trained_model, history = train_model_numeric(
        compiled_model,
        X_num=numeric_input_train,
        Y_num=y_train_train,
        batch_size=batch_size,
        patience=patience,
        validation_split=validation_split
    )

    val_mse = np.min(history.history['val_loss'])
    if target == "rating":
        val_mae = np.min(history.history['val_mae'])
    if target == "player":
        val_mae = np.min(history.history['val_loss'])

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model_name=f"model_{target}",model_type="model_num", model=trained_model)
    save_results(history=history, model_name=f"model_{target}", model_type="model_num")

    print("✅ train() done \n")

    return trained_model, val_mae, val_mse ,numeric_input_val,\
        text_input_train, text_input_val,image_input_train,image_input_val,\
        y_train_train,y_train_val

def train_text(
        text_input: np.ndarray,
        y_train: np.ndarray,
        target: str="rating",
        batch_size = 32,
        patience = 20,
        validation_split = 0.2
    ) -> float:

    max_len = MAX_Len

    file_path_tok = os.path.join(folder_path, 'tokenizer.pkl')
    with open(file_path_tok, 'rb') as f:
        tokenizer = pickle.load(f)

    input_dim=len(tokenizer.word_index)+1

    model = initialize_model_text(input_dim=input_dim,max_len=max_len)

    compiled_model = compile_model(model,target=target)

    trained_model, history = train_model_text(
        compiled_model,
        X_text=text_input,
        Y_text=y_train,
        batch_size=batch_size,
        patience=patience,
        validation_split=validation_split
    )

    val_mse = np.min(history.history['val_loss'])
    if target == "rating":
        val_mae = np.min(history.history['val_mae'])
    if target == "player":
        val_mae = np.min(history.history['val_loss'])

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model_name=f"model_{target}",model_type="model_text", model=trained_model)
    save_results(history=history, model_name=f"model_{target}", model_type="model_text")

    print("✅ train() done \n")

    return trained_model, val_mae, val_mse

def train_image(
        image_input: np.ndarray,
        y_train: np.ndarray,
        target: str="rating",
        batch_size = 32,
        patience = 20,
        validation_split = 0.2
    ) -> float:

    #images_input = np.array([load_and_preprocess_image(path).numpy() for path in image_input])


    model = initialize_cnn_model()

    compiled_model = compile_model(model,target=target)

    trained_model, history = train_model_image(
        compiled_model,
        X_image=image_input,
        Y_image=y_train,
        batch_size=batch_size,
        patience=patience,
        validation_split=validation_split
    )

    val_mse = np.min(history.history['val_loss'])
    if target == "rating":
        val_mae = np.min(history.history['val_mae'])
    if target == "player":
        val_mae = np.min(history.history['val_loss'])

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model_name=f"model_{target}",model_type="model_image", model=trained_model)
    save_results(history=history, model_name=f"model_{target}", model_type="model_image")

    print("✅ train() done \n")

    return trained_model, val_mae, val_mse

def train_meta_model(
        X_val_num: np.ndarray,
        X_val_text: np.ndarray,
        X_val_image: np.ndarray,
        y_val: np.ndarray,
        target: str="rating",
        batch_size = 32,
        patience = 20,
        validation_split = 0.2
    ) -> float:

    trained_model_num = load_most_recent_model(folder_path, model_name=f"model_{target}", model_type='model_num')
    trained_model_text = load_most_recent_model(folder_path, model_name=f"model_{target}", model_type='model_text')
    trained_model_image = load_most_recent_model(folder_path, model_name=f"model_{target}", model_type='model_image')

    preds_numeric = trained_model_num.predict(X_val_num)
    preds_text = trained_model_text.predict(X_val_text)
    preds_new_image = trained_model_image.predict(X_val_image)
    X_meta = {"base_pred_input1":preds_numeric,
                "base_pred_input2":preds_text,
                "base_pred_input3":preds_new_image}

    model = initialize_metamodel()

    compiled_model = compile_model(model,target=target)

    trained_model, history = train_metamodel(
        compiled_model,
        X_meta=X_meta,
        Y_meta=y_val,
        batch_size=batch_size,
        patience=patience,
        validation_split=validation_split
    )

    val_mse = np.min(history.history['val_loss'])
    if target == "rating":
        val_mae = np.min(history.history['val_mae'])
    if target == "player":
        val_mae = np.min(history.history['val_loss'])

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model_name=f"model_{target}",model_type="model_meta", model=trained_model)
    save_results(history=history, model_name=f"model_{target}", model_type="model_meta")

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

if __name__ == '__main__':

    # training rating model
    data_X, data_Y = get_data(target="player")

    max_len, numeric_input,text_input,image_input = preprocess(data_X[:50])

    trained_model_num, val_mae, val_mse ,numeric_input_val,\
        text_input_train, text_input_val,image_input_train,image_input_val,\
        y_train_train,y_train_val = train_numeric(
        numeric_input=numeric_input,
        text_input=text_input,
        image_input=image_input,
        y_train=data_Y[:50],
        target="player",
        batch_size = 32,
        patience = 20,
        validation_split = 0.2)

    trained_model_text, _, _ = train_text(
        text_input=text_input_train,
        y_train=y_train_train,
        target="player",
        batch_size = 32,
        patience = 20,
        validation_split = 0.2)

    trained_model_image, _, _ =train_image(
        image_input=image_input_train,
        y_train=y_train_train,
        target="player",
        batch_size = 32,
        patience = 20,
        validation_split = 0.2
    )

    trained_model_meta, _, _ = train_meta_model(
        X_val_num=numeric_input_val,
        X_val_text=text_input_val,
        X_val_image=image_input_val,
        y_val=y_train_val,
        target="player",
        batch_size = 32,
        patience = 20,
        validation_split = 0.2
    )

    #testing rating model
    data_X_test, y_test = get_test_data(target="player")

    numeric_input_test,text_input_test, image_input_test = preprocess_test(X=data_X_test)

    evaluate_model(trained_model_num,numeric_input_test,y_test,batch_size=32,target="player")
    evaluate_model(trained_model_text,text_input_test,y_test,batch_size=32,target="player")
    evaluate_model(trained_model_image,image_input_test,y_test,batch_size=32,target="player")

    preds_new_numeric = trained_model_num.predict(numeric_input_test)
    preds_new_text = trained_model_text.predict(text_input_test)
    preds_new_image = trained_model_image.predict(image_input_test)

    X_meta_test = {"base_pred_input1":preds_new_numeric,
                   "base_pred_input2":preds_new_text,
                   "base_pred_input3":preds_new_image}

    evaluate_model(trained_model_meta,X_meta_test,y_test,batch_size=32)
