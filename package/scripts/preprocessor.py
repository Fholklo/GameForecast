import numpy as np
import pandas as pd
import tensorflow as tf
import requests

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from package.scripts.params import FEATURE_SELECTION_V2,languages,days_in_year,months_in_year,genre_options,category_options
from package.scripts.params import developer_categories_dict,publishers_category_dict, european_langs

def download_image(url, app_id, index=0, folder_name='image_data', size=(256, 256)):
    try:
        # Obtenir l'image depuis l'URL
        response = requests.get(url)
        response.raise_for_status()  # Ceci va arrêter le processus en cas d'erreur

        # Convertir le contenu binaire de l'image en un objet Image et redimensionner
        image =  tf.image.decode_jpeg(response.content, channels=3)
        image = tf.image.resize(image, size)  # Redimensionner l'image en 256x256 pixels

        # Construire le chemin du fichier
        file_path = f"{folder_name}/{app_id}_{index}.jpg"

        # Écrire l'image redimensionnée dans un fichier en ajustant la qualité pour la compression
        image.save(file_path, 'JPEG')  # Réduire la qualité pour compresser l'image
        return file_path
    except Exception as e:
        print(f"Erreur lors du téléchargement de {url}: {e}")
        return None

def format_link(app_id):
    return f'package/image_data/{app_id}_0.jpg'

def clean_data(data_X:pd.DataFrame, train: bool) -> pd.DataFrame:
    '''clean the features before entering pipelines'''

    data_X = data_X[FEATURE_SELECTION_V2].copy()

    #Supported_Languages processing
    data_X.Supported_Languages.fillna('Missing',inplace=True)
    data_X['Supported_Languages'] = data_X['Supported_Languages'].apply(lambda x: x.split(', '))
    data_X["European languages"] = data_X['Supported_Languages'].apply(lambda langs: 1 \
        if any(lang in langs for lang in european_langs) else 0)
    for language in languages:
        if language == "European languages":
            pass
        data_X[language] = data_X['Supported_Languages'].apply(lambda x: 1 if language in x else 0)
    data_X['Autre_lang'] = data_X['Supported_Languages'].apply(lambda x: 1 if len(set(x) - set(languages)) > 0 else 0)
    # Ajouter des colonnes manquantes pour les genres qui ne sont pas présents dans le jeu
    missing_languages = list(set(languages) - set(data_X.columns))
    for language in missing_languages:
        data_X[language] = 0

    data_X['Release_Date'] = pd.to_datetime(data_X['Release_Date'])

    data_X['day_sin'] = np.sin(2 * np.pi * data_X['Release_Date'].dt.dayofyear / days_in_year)
    data_X['day_cos'] = np.cos(2 * np.pi * data_X['Release_Date'].dt.dayofyear / days_in_year)
    data_X['month_sin'] = np.sin(2 * np.pi * data_X['Release_Date'].dt.month / months_in_year)
    data_X['month_cos'] = np.cos(2 * np.pi * data_X['Release_Date'].dt.month / months_in_year)
    data_X['year'] = data_X['Release_Date'].dt.year
    data_X.drop(columns="Release_Date",inplace=True)

    #catégories pour dévelopers
    data_X['Developers'] = data_X['Developers'].fillna(-1)
    def get_category_by_dev(developer_name):
        return developer_categories_dict.get(developer_name, -1)
    data_X['Developers'] = data_X['Developers'].apply(get_category_by_dev)
    data_X['Developers'] = data_X['Developers'].astype(float)

    #catégories pour publisher
    data_X['Publishers'] = data_X['Publishers'].fillna(-1)
    def get_category_by_pub(publisher_name):
        return publishers_category_dict.get(publisher_name, -1)
    data_X['Publishers'] = data_X['Publishers'].apply(get_category_by_pub)
    data_X['Publishers'] = data_X['Publishers'].astype(float)

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

    #data_X = data_X[data_X.Price != 'None']
    data_X.Price.replace("None",0,inplace=True)
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

    data_X.Categories.fillna('Missing',inplace=True)
    data_X['Categories'] = data_X['Categories'].apply(lambda x: x.split(','))
    for categorie in category_options:
        data_X[categorie] = data_X['Categories'].apply(lambda x: 1 if categorie in x else 0)
    data_X['Autre_cat'] = data_X['Categories'].apply(lambda x: 1 if len(set(x) - set(category_options)) > 0 else 0)
    # Ajouter des colonnes manquantes pour les genres qui ne sont pas présents dans le jeu
    missing_categories = list(set(category_options) - set(data_X.columns))
    for categorie in missing_categories:
        data_X[categorie] = 0

    if train:
        # This just copies chemins_images, could use directly
        data_X["Screenshots"] = data_X["App_ID"].apply(format_link)
    else:
        download_image(url=data_X["Screenshots"],app_id=data_X["App_ID"])
        data_X["Screenshots"] = data_X["App_ID"].apply(format_link)

    data_X.drop(columns=["App_ID","Categories","Genres","Supported_Languages"],inplace = True)

    data_X["About_The_Game"].fillna(value="Missing",inplace=True)

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















# def clean_target(data:pd.DataFrame) -> pd.DataFrame :
#     '''clean and cut the target'''
#     data = data[data['Month'] != 'Last 30 Days']
#     data['Month'] = pd.to_datetime(data['Month'])
#     counts = data['App_ID'].value_counts()
#     data = data[data['App_ID'].isin(counts[counts > 1].index)]
#     data = data[(data['Month'] >= '2012-07-01') & (data['Month'] <= '2024-01-31')]

#     return data

# def only_last_month_v1_target(data:pd.DataFrame) -> pd.DataFrame :
#     '''V1 : select only the last 2 month to predict the avg # of players'''

#     data = data.groupby('App_ID',sort=False).last().reset_index()
#     data.drop(columns='Month',inplace=True)
#     return data

# def transform_language_features(data_X: pd.DataFrame) -> pd.DataFrame:
#     assert isinstance(data_X, pd.DataFrame)

#     unique_languages = [lang.title() for lang in UNIQUE_LANGUAGE]
#     # initialize lists to store languages and their proportions
#     language_proportions = {}

#     # calculate proportion for each language
#     for lang in unique_languages:
#         lang_count = (data_X['Supported_Languages'].str.contains(lang).sum()) / len(data_X)
#         language_proportions[lang] = lang_count

#     # order depending on language proportions
#     sorted_languages = sorted(language_proportions.items(), key=lambda x: x[1], reverse=True)

#     # define european language
#     language_proportions['European'] = language_proportions['German'] + language_proportions['French'] + language_proportions['Italian'] + language_proportions['Spanish - Spain'] + language_proportions['Portuguese - Portugal']

#     # Initialize top languages
#     top_languages = []

#     # iterate over the languages to add them to the list
#     for lang, proportion in sorted_languages:
#         if lang not in EUROPEAN_LANGUAGES:
#             top_languages.append(lang)
#         if len(top_languages) == TOP_LANGUAGES:
#             break

#     # make every language a colomn
#     for lang in top_languages:
#         if lang == 'European':  # Utilisez '==' pour la comparaison d'égalité, pas '='
#             data_X[lang] = data_X['Supported_Languages'].str.contains("German|French|Italian|Spanish - Spain|Portuguese - Portugal", case=False, regex=True)
#         else:
#             data_X[lang] = data_X['Supported_Languages'].str.contains(lang, case=False, regex=True)

#         data_X[lang] = data_X[lang].astype(int)

#     data_X['other_lang'] = ~data_X['Supported_Languages'].str.contains('|'.join(top_languages), case=False, regex=True)
#     data_X['other_lang'] = data_X['other_lang'].astype(int)

#     data_X = data_X[data_X['English'] != 0]

#     return data_X
