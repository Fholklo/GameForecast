from fastapi import FastAPI
import pandas as pd
from tensorflow import keras

#from package.ml_logics.registry import *
from package.main import preprocess

app = FastAPI()
#app.state.model_rating = load_model("model_rating")
app.state.model_rating =keras.models.load_model("model_rating_20240320-232349.h5")
#app.state.model_player_r = load_model("model_player_r")

@app.get("/")
def root():
    return {'greetings':"hello"}

@app.get("/predict_rating")
def predict_rating(
    supported_languages: str, # English, French, etc..
    support_url: str, # url du support
    developers: str, #ankama
    publishers: str, #bob
    release_date: str, #"2024-01-31"
    genres: str, #["genre1","genre2","genre3"]
    categorie: str, #["categories1","categories2"]
    windows: bool, #true
    mac: bool, #false
    linux: bool, #false
    achievements: int,#40
    price: float#2.98
    ):

    X_pred = pd.DataFrame(dict(
        supported_languages = [supported_languages],
        support_url= [support_url],
        developers= [developers],
        publishers= [publishers],
        categories= [categorie],
        release_date= [release_date],
        genres= [genres],
        categorie= [categorie],
        windows= [windows],
        mac= [mac],
        linux= [linux],
        achievements= [achievements],
        price= [price],
        app_id = 0,
        positive = 0,
        negative = 0
    ))
    X_pred.rename(columns={
    'supported_languages': 'Supported_Languages',
    'support_url': 'Support_URL',
    'developers': 'Developers',
    'publishers': 'Publishers',
    'categories': 'Categories',
    'release_date': 'Release_Date',
    'genres': 'Genres',
    'windows': 'Windows',
    'mac': 'Mac',
    'linux': 'Linux',
    'achievements': 'Achievements',
    'price': 'Price',
    "app_id": "App_ID",
    "positive":"Positive",
    "negative": "Negative"
}, inplace=True)

    X_pred_preprocess = preprocess(X_pred)
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

    X_pred_preprocess, _ = preprocess(X_pred)
    y_pred = app.state.model_player_r.predict(X_pred_preprocess)

    return {"Peak player" : float(y_pred)}
