import numpy as np
import pandas as pd
import string
from datetime import datetime

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

from scripts.params import *

def clean_target(data):
    '''clean and cut the target'''
    data = data[data['Month'] != 'Last 30 Days']
    data['Month'] = pd.to_datetime(data['Month'])
    counts = data['App_ID'].value_counts()
    data = data[data['App_ID'].isin(counts[counts > 1].index)]
    data = data[(data.iloc[:, 1].apply(lambda x: x.month) >= 7) & \
                (data.iloc[:, 1].apply(lambda x: x.year) >= 2012)]

    return data

def basic_cleaning(sentence):
    '''preprocess a basic cleaning for text of raw data'''
    sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '  ')
    sentence = sentence.strip()
    sentence = sentence.split("  ")
    sentence = [word for word in sentence if word != ""]

    return sentence

def transform_language_features(data_X: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(data_X, pd.DataFrame)

    # initialize lists to store languages and their proportions
    language_proportions = {}

    # calculate proportion for each language
    for lang in UNIQUE_LANGUAGES:
        lang_count = (data_X['Supported_Languages'].str.contains(lang).sum()) / len(data_X)
        language_proportions[lang] = lang_count

    # order depending on language proportions
    sorted_languages = sorted(language_proportions.items(), key=lambda x: x[1], reverse=True)

    # define european language
    language_proportions['European'] = language_proportions['German'] + language_proportions['French'] + language_proportions['Italian'] + language_proportions['Spanish - Spain'] + language_proportions['Portuguese - Portugal']

    # Initialize top languages
    top_languages = []

    # iterate over the languages to add them to the list
    for lang, proportion in sorted_languages:
        if lang not in EUROPEAN_LANGUAGES:
            top_languages.append(lang)
        if len(top_languages) == 10:
            break

    # make every language a colomn
    for lang in top_languages:
        if lang == 'European':  # Utilisez '==' pour la comparaison d'égalité, pas '='
            data_X[lang] = data_X['Supported_Languages'].str.contains("German|French|Italian|Spanish - Spain|Portuguese - Portugal", case=False, regex=True)
        else:
            data_X[lang] = data_X['Supported_Languages'].str.contains(lang, case=False, regex=True)

        data_X[lang] = data_X[lang].astype(int)

    return data_X

def clean_data(data_X,data_Y):
    '''clean the features before entering pipelines'''
    Y = clean_target(data_Y)
    # consistent features - target
    data_X = data_X[data_X['App_ID'].isin(Y['App_ID'])]
    data_X['Release_Date'] = pd.to_datetime(data_X['Release_Date'])
    # basic sentence cleaning
    data_X.Supported_Languages = data_X.Supported_Languages.apply(basic_cleaning)
    # keep only games with at least english language
    data_X = data_X[data_X['Supported_Languages'].apply(lambda x: 'english' in x)]
    # transform support url with 1 if contains something, 0 otherwise
    data_X.Support_URL = data_X['Support_URL'].apply(lambda x: 0 if x!=x else 1)
    # encode bool values
    data_X.Windows = data_X.Windows.apply(lambda x: 1 if x==True else 0)
    data_X.Linux = data_X.Linux.apply(lambda x: 1 if x==True else 0)
    data_X.Mac = data_X.Mac.apply(lambda x: 1 if x==True else 0)
    # handle categorical columns before encoding
    data_X.Genres.fillna('No',inplace=True)
    data_X.Genres = data_X.Genres.apply(lambda x: ''.join(x).split(','))
    data_X.Categories.fillna('No', inplace=True)
    data_X.Categories = data_X.Categories.apply(lambda x: ''.join(x).split(','))
    # handle numerical columns before encoding
    data_X.loc[:, 'Achievements'] = data_X['Achievements'].fillna(0)

    return data_X, Y

def full_preprocessor(data_X,data_Y):
    """Create a pipeline to preprocess data"""
    data_X, data_y = clean_data(data_X,data_Y)
    data_X = transform_language_features(data_X)
    # numerical features
    robust_features = ["Price", "Achievements"]
    # numerical pipeline
    scalers = ColumnTransformer([
        ("rob", RobustScaler(), robust_features), # Robust
    ])

    numerical_pipeline = Pipeline([
        ("imputer", KNNImputer()),
        ("scalers", scalers)
    ])
    # categorical features
    onehot_features = ["Genres", "categories", "Developpers", "Publishers"]
    # categorical pipeline
    encoders = ColumnTransformer([
        ("one_hot", OneHotEncoder(sparse_output=False,
                                drop="if_binary",
                                handle_unknown="ignore"),
        onehot_features) # OHE
    ], remainder="passthrough")

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant")),
        ("encoders", encoders)
    ])
    # Full_preprocessor
    preprocessor = ColumnTransformer([
        ("num_pipeline", numerical_pipeline, make_column_selector(dtype_include="number")), # num_features
        ("cat_pipeline", categorical_pipeline, make_column_selector(dtype_exclude="number")) # cat_features
    ], remainder="passthrough").set_output(transform="pandas")


    return preprocessor
