from fastapi import FastAPI
import pandas as pd
from tensorflow import keras

#from package.ml_logics.registry import *
from package.main import preprocess

app = FastAPI()
#app.state.model_rating = load_model("model_rating")
app.state.model_rating =keras.models.load_model("model_rating_20240321-102424.h5")
#app.state.model_player_r = load_model("model_player_r")

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
    ))
    preprocessor, X_pred_preprocess = preprocess(X_pred)
    y_pred = app.state.model_rating.predict(X_pred_preprocess)

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

    preprocessor, X_pred_preprocess, _ = preprocess(X_pred)
    y_pred = app.state.model_player.predict(X_pred_preprocess)

    return {"Peak player" : float(y_pred)}
