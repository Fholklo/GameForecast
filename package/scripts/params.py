import os
import json

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")

GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")

############## CONSTANTS ###################
#Path
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "gameforecast", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "gameforecast", "training_outputs")

#################
### MASTER
#################
with open('package/param_json/developer_categories_dict.json', 'r') as json_file:
    developer_categories_dict = json.load(json_file)

with open('package/param_json/publisher_categories_dict.json', 'r') as json_file2:
    publishers_category_dict = json.load(json_file2)

#################
### LOCAL NOTEBOOKS
#################
#with open('../package/param_json/developer_categories_dict.json', 'r') as json_file:
#    developer_categories_dict = json.load(json_file)

#with open('../package/param_json/publisher_categories_dict.json', 'r') as json_file2:
#    publishers_category_dict = json.load(json_file2)

#Preprocessing

FEATURE_SELECTION_V2 = ["App_ID","Release_Date","Price","About_The_Game","Supported_Languages","Support_URL","Windows","Mac","Linux","Achievements","Developers","Publishers","Categories","Genres","Screenshots"]

genre_options = ['Accounting', 'Action', 'Adventure', 'Animation & Modeling', 'Autre',
                 'Audio Production', 'Casual', 'Design & Illustration', 'Early Access',
                 'Education', 'Free to Play', 'Game Development', 'Gore', 'Indie',
                 'Massively Multiplayer', 'Movie', 'Nudity', 'Photo Editing', 'RPG',
                 'Racing', 'Sexual Content', 'Simulation', 'Software Training', 'Sports',
                 'Strategy', 'Utilities', 'Video Production', 'Violent','Web Publishing','Missing']

category_options = ['Single-player', 'Family Sharing', 'Steam Achievements', 'Steam Cloud',
                    'Steam Trading Cards', 'Full controller support', 'Multi-player', 'PvP',
                    'Partial Controller Support', 'Co-op', 'Online PvP', 'Steam Leaderboards',
                    'Remote Play Together', 'Online Co-op', 'Shared/Split Screen',
                    'Remote Play on TV', 'Steam Workshop', 'Stats', 'Shared/Split Screen PvP',
                    'Cross-Platform Multiplayer', 'Tracked Controller Support',
                    'Shared/Split Screen Co-op', 'Includes level editor', 'In-App Purchases',
                    'Remote Play on Tablet', 'VR Only', 'Remote Play on Phone', 'Captions available',
                    'MMO', 'VR Supported', 'Missing', 'LAN Co-op', 'LAN PvP']

languages = [
    "English",               # Anglais
    "European languages"     # French, German, Italian, Portuguese
    "Simplified Chinese",    # Chinois simplifié
    "Russian",               # Russe
    "Japanese",              # Japonais
    "Korean",                # Coréen
    "Traditional Chinese",   # Chinois traditionnel
    "Portuguese - Brazil",   # Portugais - Brésil
    "Polish",                # Polonais
    "Turkish",               # Turc
    "Spanish - Latin America", # Espagnol - Amérique latine
    "other_lang"             # Autre langue (remplacez "other_lang" par le nom spécifique si vous le connaissez)
]

european_langs = ["French", "German", "Italian", "Portuguese - Portugal", "Spanish - Spain"]

days_in_year = 365.25  # moyenne en tenant compte des années bissextiles

months_in_year = 12

MAX_Len = 2862

#main
folder_path = 'package/tok_preproc_model'
params_file_path = 'package/scripts/max_len.py'
