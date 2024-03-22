import pandas as pd
from datetime import datetime
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector


from package.scripts.params import *

def clean_target(data:pd.DataFrame) -> pd.DataFrame :
    '''clean and cut the target'''
    data = data[data['Month'] != 'Last 30 Days']
    data['Month'] = pd.to_datetime(data['Month'])
    counts = data['App_ID'].value_counts()
    data = data[data['App_ID'].isin(counts[counts > 1].index)]
    data = data[(data['Month'] >= '2012-07-01') & (data['Month'] <= '2024-01-31')]

    return data

def only_last_month_v1_target(data:pd.DataFrame) -> pd.DataFrame :
    '''V1 : select only the last 2 month to predict the avg # of players'''

    data = data.groupby('App_ID',sort=False).last().reset_index()
    data.drop(columns='Month',inplace=True)
    return data

def transform_language_features(data_X: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(data_X, pd.DataFrame)

    unique_languages = [lang.title() for lang in UNIQUE_LANGUAGE]
    # initialize lists to store languages and their proportions
    language_proportions = {}

    # calculate proportion for each language
    for lang in unique_languages:
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
        if len(top_languages) == TOP_LANGUAGES:
            break

    # make every language a colomn
    for lang in top_languages:
        if lang == 'European':  # Utilisez '==' pour la comparaison d'égalité, pas '='
            data_X[lang] = data_X['Supported_Languages'].str.contains("German|French|Italian|Spanish - Spain|Portuguese - Portugal", case=False, regex=True)
        else:
            data_X[lang] = data_X['Supported_Languages'].str.contains(lang, case=False, regex=True)

        data_X[lang] = data_X[lang].astype(int)

    data_X['other_lang'] = ~data_X['Supported_Languages'].str.contains('|'.join(top_languages), case=False, regex=True)
    data_X['other_lang'] = data_X['other_lang'].astype(int)

    data_X = data_X[data_X['English'] != 0]

    return data_X


def clean_data(data_X:pd.DataFrame) :
    '''clean the features before entering pipelines'''

    data_X = data_X[FEATURE_SELECTION_V1].copy()


    #Supported_Languages processing
    data_X.Supported_Languages.fillna('Missing',inplace=True)
    data_X['Supported_Languages'] = data_X['Supported_Languages'].apply(lambda x: x.split(', '))
    for language in languages:
        data_X[language] = data_X['Supported_Languages'].apply(lambda x: 1 if language in x else 0)
    data_X['Autre_lang'] = data_X['Supported_Languages'].apply(lambda x: 1 if len(set(x) - set(languages)) > 0 else 0)
    # Ajouter des colonnes manquantes pour les genres qui ne sont pas présents dans le jeu
    missing_languages = list(set(languages) - set(data_X.columns))
    for language in missing_languages:
        data_X[language] = 0

    data_X['Release_Date'] = pd.to_datetime(data_X['Release_Date'], format="mixed")

    data_X['day_sin'] = np.sin(2 * np.pi * data_X['Release_Date'].dt.dayofyear / days_in_year)
    data_X['day_cos'] = np.cos(2 * np.pi * data_X['Release_Date'].dt.dayofyear / days_in_year)
    data_X['month_sin'] = np.sin(2 * np.pi * data_X['Release_Date'].dt.month / months_in_year)
    data_X['month_cos'] = np.cos(2 * np.pi * data_X['Release_Date'].dt.month / months_in_year)
    data_X['year'] = data_X['Release_Date'].dt.year
    data_X.drop(columns="Release_Date",inplace=True)

    # keep only games with at least english language
    #data_X = transform_language_features(data_X)

    #catégories pour dévelopers
    data_X['Developers'] = data_X['Developers'].fillna(-1)
    def get_category_by_dev(developer_name):
        return developer_categories_dict.get(developer_name, -1)
    data_X['Developers'] = data_X['Developers'].apply(get_category_by_dev)
    data_X['Developers'] = data_X['Developers'].astype(float)

    #developer_counts = data_X['Developers'].groupby(data_X['Developers']).transform('count')
    #data_X["dev_category"] = developer_counts.apply(assign_category_developer)
    #data_X.drop(columns="Developers", inplace=True)

    #catégories pour publisher
    data_X['Publishers'] = data_X['Publishers'].fillna(-1)
    def get_category_by_pub(publisher_name):
        return publishers_category_dict.get(publisher_name, -1)
    data_X['Publishers'] = data_X['Publishers'].apply(get_category_by_pub)
    data_X['Publishers'] = data_X['Publishers'].astype(float)

    #developer_counts = data_X['Publishers'].groupby(data_X['Publishers']).transform('count')
    #data_X["publi_category"] = developer_counts.apply(assign_category_developer)
    #data_X.drop(columns="Publishers", inplace=True)

    # transform support url with 1 if contains something, 0 otherwise
    data_X.Support_URL = data_X['Support_URL'].apply(lambda x: 0 if x!=x else 1)
    # encode bool values
    data_X.Windows = data_X.Windows.apply(lambda x: 1 if x==True else 0)
    data_X.Linux = data_X.Linux.apply(lambda x: 1 if x==True else 0)
    data_X.Mac = data_X.Mac.apply(lambda x: 1 if x==True else 0)

    # handle numerical columns before encoding
    data_X.loc[:, 'Achievements'] = data_X['Achievements'].fillna(0)
    data_X.Achievements.replace('None',0,inplace=True)
    data_X.Achievements = data_X.Achievements.astype(dtype='int64')

    data_X = data_X[data_X.Price != 'None']
    data_X.Price = data_X.Price.astype(dtype='float64')

    # handle categorical columns before encoding
    data_X.Genres.fillna('Missing',inplace=True)
    data_X['Genres'] = data_X['Genres'].apply(lambda x: x.split(','))
    for genre in genre_options:
        data_X[genre] = data_X['Genres'].apply(lambda x: 1 if genre in x else 0)
    data_X['Autre_genre'] = data_X['Genres'].apply(lambda x: 1 if len(set(x) - set(genre_options)) > 0 else 0)
    # Ajouter des colonnes manquantes pour les genres qui ne sont pas présents dans le jeu
    missing_genres = list(set(genre_options) - set(data_X.columns))
    for genre in missing_genres:
        data_X[genre] = 0

    #data_X.Genres = data_X.Genres.apply(lambda x: ''.join(x).split(','))
    #exploded_data = data_X.Genres.explode()
    #one_hot_encoded_df = pd.get_dummies(exploded_data).groupby(level=0).sum()
    #data_X = pd.concat([data_X,one_hot_encoded_df],axis=1)
    #data_X.drop(columns='Genres',inplace=True)

    data_X.Categories.fillna('Missing',inplace=True)
    data_X['Categories'] = data_X['Categories'].apply(lambda x: x.split(','))
    for categorie in category_options:
        data_X[categorie] = data_X['Categories'].apply(lambda x: 1 if categorie in x else 0)
    data_X['Autre_cat'] = data_X['Categories'].apply(lambda x: 1 if len(set(x) - set(category_options)) > 0 else 0)
    # Ajouter des colonnes manquantes pour les genres qui ne sont pas présents dans le jeu
    missing_categories = list(set(category_options) - set(data_X.columns))
    for categorie in missing_categories:
        data_X[categorie] = 0

    #data_X.Categories.fillna('No', inplace=True)
    #data_X.Categories = data_X.Categories.apply(lambda x: ''.join(x).split(','))
    #categories = data_X.Categories.explode().value_counts()/len(data_X.Categories) > percent_categories
    #true_categories = categories[categories].index.tolist()
    #data_X['autre_cat'] = data_X.Categories.apply(lambda x: [c for c in x if c not in true_categories])
    # Filtrer les catégories pour encoder seulement celles présentes dans true_categories
    #data_X['Categories'] = data_X.Categories.apply(lambda x: [c for c in x if c in true_categories])
    # Encoder les catégories
    #exploded_data = data_X.Categories.explode()
    #one_hot_encoded_df = pd.get_dummies(exploded_data).groupby(level=0).sum()
    #mise en forme de "autre"
    #data_X["autre_cat"] = data_X["autre_cat"].apply(lambda x: 1 if x else 0)


    # Ajouter les catégories encodées à data_X
    #data_X = pd.concat([data_X, one_hot_encoded_df], axis=1)
    data_X.drop(columns=["Categories","Genres","Supported_Languages"],inplace = True)

    #data_X.sort_values(by='App_ID',inplace=True)
    #Y_rating.sort_values(by='App_ID',inplace=True)
    #y.sort_values(by='App_ID',inplace=True)

    #data_X.reset_index(drop=True,inplace=True)
    #Y_rating.reset_index(drop=True,inplace=True)
    #y.reset_index(drop=True,inplace=True)

    #Y_rating.drop(columns="App_ID",inplace=True)
    #y.drop(columns="App_ID",inplace=True)

    return data_X

def full_preprocessor():
    """Create a pipeline to preprocess data"""

    # numerical features
    robust_features = ["Price", "Achievements","year"]
    # numerical pipeline
    scalers = ColumnTransformer([
        ("rob", RobustScaler(), robust_features), # Robust
    ], remainder="passthrough")

    numerical_pipeline = Pipeline([
        ("imputer", KNNImputer()),
        ("scalers", scalers)
    ])
    # categorical features
    ordinal_features = ["Developers", "Publishers"]
    # categorical pipeline
    encoders = ColumnTransformer([
        ("ordinal",OrdinalEncoder(categories="auto", handle_unknown="use_encoded_value",unknown_value=-1)
         ,ordinal_features)
    ], remainder="passthrough")

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant")),
        ("encoders", encoders)
    ])
    # Full_preprocessor
    preprocessor = ColumnTransformer([
        ("num_pipeline", numerical_pipeline, ["Price", "Achievements","year"]), # num_features # type: ignore
        ("cat_pipeline", categorical_pipeline, ['Developers', 'Publishers']) # cat_features # type: ignore
    ], remainder="passthrough").set_output(transform="pandas")


    return preprocessor
