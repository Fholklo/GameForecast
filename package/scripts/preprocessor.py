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

def assign_category_developer(count:int)-> int:
    if count == 0:
        return "0"
    elif count == 1:
        return "1"
    elif count == 2:
        return "2"
    elif count == 3:
        return "3"
    elif count == 4:
        return "4"
    elif count == 5:
        return "5"
    elif 6 <= count <= 10:
        return "6"
    elif 11 <= count <= 20:
        return "7"
    else:  # Plus de 20
        return "8"

def assign_category_publisher(count:int)-> int:
    if count == 0:
        return "0"
    elif count == 1:
        return "1"
    elif count == 2:
        return "2"
    elif count == 3:
        return "3"
    elif count == 4:
        return "4"
    elif count == 5:
        return "5"
    elif 6 <= count <= 10:
        return "6"
    else:  # Plus de 10
        return "7"

def clean_data(data_X:pd.DataFrame,data_Y:pd.DataFrame) :
    '''clean the features before entering pipelines'''

    Y_clean = clean_target(data_Y)
    y = only_last_month_v1_target(Y_clean)

    data_X = data_X[FEATURE_SELECTION_V1]

    # consistent features - target
    data_X = data_X[data_X['App_ID'].isin(y['App_ID'])]
    data_X['Release_Date'] = pd.to_datetime(data_X['Release_Date'])

    # keep only games with at least english language
    data_X = transform_language_features(data_X)

    #catégories pour dévelopers
    developer_counts = data_X['Developers'].groupby(data_X['Developers']).transform('count')
    data_X["dev_category"] = developer_counts.apply(assign_category_developer)
    data_X.drop(columns="Developers", inplace=True)

    #catégories pour publisher
    developer_counts = data_X['Publishers'].groupby(data_X['Publishers']).transform('count')
    data_X["publi_category"] = developer_counts.apply(assign_category_developer)
    data_X.drop(columns="Publishers", inplace=True)

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

    # Compute Rating for Y_rating target
    data_X['TotalReviews'] = data_X['Positive'] + data_X['Negative']
    data_X['ReviewScore'] = data_X['Positive'] / data_X['TotalReviews']
    data_X['Rating'] = data_X['ReviewScore'] - (data_X['ReviewScore'] - 0.5) * 2 ** (- np.log10(data_X['TotalReviews']) + 1)
    data_X.Rating.fillna(0,inplace=True)

    Y_rating = data_X[['App_ID','Rating']]
    data_X.drop(columns=['TotalReviews', 'ReviewScore','Positive','Negative','Supported_Languages',"Rating"],inplace=True)

    data_X = data_X[data_X.Price != 'None']
    data_X.Price = data_X.Price.astype(dtype='float64')

    data_X.Achievements.replace('None',0,inplace=True)
    data_X.Achievements = data_X.Achievements.astype(dtype='int64')

    exploded_data = data_X.Genres.explode()
    one_hot_encoded_df = pd.get_dummies(exploded_data).groupby(level=0).sum()
    data_X = pd.concat([data_X,one_hot_encoded_df],axis=1)
    data_X.drop(columns='Genres',inplace=True)

    categories = data_X.Categories.explode().value_counts()/len(data_X.Categories) > percent_categories
    true_categories = categories[categories].index.tolist()
    data_X['autre_cat'] = data_X.Categories.apply(lambda x: [c for c in x if c not in true_categories])
    # Filtrer les catégories pour encoder seulement celles présentes dans true_categories
    data_X['Categories'] = data_X.Categories.apply(lambda x: [c for c in x if c in true_categories])
    # Encoder les catégories
    exploded_data = data_X.Categories.explode()
    one_hot_encoded_df = pd.get_dummies(exploded_data).groupby(level=0).sum()

    #mise en forme de "autre"
    data_X["autre_cat"] = data_X["autre_cat"].apply(lambda x: 1 if x else 0)

    # Ajouter les catégories encodées à data_X
    data_X = pd.concat([data_X, one_hot_encoded_df], axis=1)
    data_X.drop(columns=["Categories"],inplace = True)

    data_X.sort_values(by='App_ID',inplace=True)
    Y_rating.sort_values(by='App_ID',inplace=True)
    y.sort_values(by='App_ID',inplace=True)

    y = y[y['App_ID'].isin(data_X['App_ID'])]

    data_X.reset_index(drop=True,inplace=True)
    Y_rating.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    data_X.drop(columns="App_ID",inplace=True)
    Y_rating.drop(columns="App_ID",inplace=True)
    y.drop(columns="App_ID",inplace=True)

    return data_X, Y_rating, y

def full_preprocessor():
    """Create a pipeline to preprocess data"""

    # numerical features
    robust_features = ["Price", "Achievements"]
    # numerical pipeline
    scalers = ColumnTransformer([
        ("rob", RobustScaler(), robust_features), # Robust
    ], remainder="passthrough")

    numerical_pipeline = Pipeline([
        ("imputer", KNNImputer()),
        ("scalers", scalers)
    ])
    # categorical features
    ordinal_features = ["publi_category", "dev_category"]
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
        ("num_pipeline", numerical_pipeline, make_column_selector(dtype_include="number")), # num_features # type: ignore
        ("cat_pipeline", categorical_pipeline, make_column_selector(dtype_exclude="number")) # cat_features # type: ignore
    ], remainder="passthrough").set_output(transform="pandas")


    return preprocessor
