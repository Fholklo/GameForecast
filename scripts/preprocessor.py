import numpy as np
import pandas as pd
from datetime import datetime

from scripts.params import *

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

    df = data.groupby('App_ID',sort=False).last().reset_index()
    df.drop(columns='Month',inplace=True)
    return df

def basic_cleaning(sentence:str) -> str :
    '''preprocess a basic cleaning for text in raw data'''

    sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())
    for punctuation in ["[","]","'","#","\\r\\n"]:
        sentence = sentence.replace(punctuation, '  ')
    sentence = sentence.strip()
    sentence = sentence.split("  ")
    sentence = [word.strip() for word in sentence]
    sentence = [word for word in sentence if word != ""]

    return sentence

def clean_data(data_X:pd.DataFrame,data_Y:pd.DataFrame) -> pd.DataFrame:
    '''clean the features before entering pipelines'''

    Y = clean_target(data_Y)
    Y = only_last_month_v1_target(Y)

    data_X = data_X[FEATURE_SELECTION_V1]

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

    # Compute Rating for Y_rating target
    data_X['TotalReviews'] = data_X['Positive'] + data_X['Negative']
    data_X['ReviewScore'] = data_X['Positive'] / data_X['TotalReviews']
    data_X['Rating'] = data_X['ReviewScore'] - (data_X['ReviewScore'] - 0.5) * 2 ** (- np.log10(data_X['TotalReviews']) + 1)
    data_X.Rating.fillna(0,inplace=True)

    Y_rating = data_X[['App_ID','Rating']]
    data_X.drop(columns=['TotalReviews', 'ReviewScore','Positive','Negative'],inplace=True)

    return data_X, Y_rating, Y
