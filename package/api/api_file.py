from fastapi import FastAPI
import pandas as pd
from tensorflow import keras

from package.main import preprocess_test, load_most_recent_model
from package.front_end.params_acc import folder_path_player,folder_path_rating

app = FastAPI()

app.state.model_rating_num =keras.models.load_model("")
app.state.model_rating_text =keras.models.load_model("")
app.state.model_rating_image =keras.models.load_model("")
app.state.model_rating_metamodel =keras.models.load_model("")

app.state.model_player_num =keras.models.load_model("")
app.state.model_player_text =keras.models.load_model("")
app.state.model_player_image =keras.models.load_model("")
app.state.model_player_metamodel =keras.models.load_model("")

@app.get("/")
def root():
    return {'greetings':"hello"}

@app.get("/predict_rating")
def predict_rating(
    Supported_Languages: str, # English, French, etc..
    Support_URL: str, # url du support
    Developers: str, #ankama
    Publishers: str, #bob
    Release_Date: str, #"2024-01-31"
    Genres: str, #["genre1","genre2","genre3"]
    Categories: str, #["categories1","categories2"]
    Windows: bool, #true
    Mac: bool, #false
    Linux: bool, #false
    Achievements: int,#40
    Price: float#2.98
    ):

    X_pred = pd.DataFrame(dict(
        Supported_Languages = [Supported_Languages],
        Support_URL= [Support_URL],
        Developers= [Developers],
        Publishers= [Publishers],
        Release_Date= [Release_Date],
        Genres= [Genres],
        Categories= [Categories],
        Windows= [Windows],
        Mac= [Mac],
        Linux= [Linux],
        Achievements= [Achievements],
        Price= [Price],
        About_The_Game = [About_The_game],
        Screenshots = [Screenshosts]
    ))
    numeric_input,text_input, images_input = preprocess_test(X_pred)

    trained_model_num = load_most_recent_model(folder_path_rating, 'model_num')
    trained_model_text = load_most_recent_model(folder_path_rating, 'model_text')
    trained_model_image = load_most_recent_model(folder_path_rating, 'model_image')

    preds_new_numeric = trained_model_num.predict(numeric_input)
    preds_new_text = trained_model_text.predict(text_input)
    preds_new_image = trained_model_image.predict(images_input)

    X_meta_test = {"base_pred_input1":preds_new_numeric,
                   "base_pred_input2":preds_new_text,
                   "base_pred_input3":preds_new_image}


    y_pred = app.state.model_rating_metamodel.predict(X_meta_test)

    return {"Rating" : float(y_pred)}

@app.get("/predict_player_release")

def predict_player_release(
    App_ID: int, #1546456
    supported_languages: str, # English, French, etc..
    support_url: str, # url du support
    developers: str, #ankama
    publishers: str, #bob
    categories: str, #Single-player
    release_date: str, #"2024-01-31"
    genres: list, #["genre1","genre2","genre3"]
    categorie: list, #["categories1","categories2"]
    windows: bool, #true
    mac: bool, #false
    linux: bool, #false
    achievements: int,#40
    price: float#2.98
    ):

    X_pred = pd.DataFrame(dict(
        App_ID = [App_ID],
        supported_languages = [supported_languages],
        support_url= [support_url],
        developers= [developers],
        publishers= [publishers],
        categories= [categories],
        release_date= [release_date],
        genres= [genres],
        categorie= [categorie],
        windows= [windows],
        mac= [mac],
        linux= [linux],
        achievements= [achievements],
        price= [price]
    ))

    numeric_input,text_input, images_input = preprocess_test(X_pred)

    trained_model_num = load_most_recent_model(folder_path_player, 'model_num')
    trained_model_text = load_most_recent_model(folder_path_player, 'model_text')
    trained_model_image = load_most_recent_model(folder_path_player, 'model_image')

    preds_new_numeric = trained_model_num.predict(numeric_input)
    preds_new_text = trained_model_text.predict(text_input)
    preds_new_image = trained_model_image.predict(images_input)

    X_meta_test = {"base_pred_input1":preds_new_numeric,
                   "base_pred_input2":preds_new_text,
                   "base_pred_input3":preds_new_image}


    y_pred = app.state.model_rating_metamodel.predict(X_meta_test)

    return {"Peak player" : float(y_pred)}
