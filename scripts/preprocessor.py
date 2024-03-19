import numpy as np
import pandas as pd
import string
from datetime import datetime

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

    return data_X, Y
