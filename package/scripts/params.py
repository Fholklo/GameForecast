import os
import json

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")

GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")


############## CONSTANTS ###################
#Path
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "gameforecast", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "gameforecast", "training_outputs")

#Data

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

# Catégories: nb de jeu par dévelopeur
#     if count == 0:
#         return "0"
#     elif count == 1:
#         return "1"
#     elif count == 2:
#         return "2"
#     elif count == 3:
#         return "3"
#     elif count == 4:
#         return "4"
#     elif count == 5:
#         return "5"
#     elif 6 <= count <= 10:
#         return "6"
#     elif 11 <= count <= 20:
#         return "7"
#     else:  # Plus de 20
#         return "8"

# Catégories: nb de jeu par publisher
#     if count == 0:
#         return "0"
#     elif count == 1:
#         return "1"
#     elif count == 2:
#         return "2"
#     elif count == 3:
#         return "3"
#     elif count == 4:
#         return "4"
#     elif count == 5:
#         return "5"
#     elif 6 <= count <= 10:
#         return "6"
#     else:  # Plus de 10
#         return "7"


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

required_fields = ['App_ID', 'Developers', 'Publishers', 'Achievements', 'Price']


#Preprocess
FEATURE_SELECTION_V1 = ["Release_Date","Price","Supported_Languages","Support_URL","Windows","Mac","Linux","Achievements","Developers","Publishers","Categories","Genres"]
FEATURE_SELECTION_V2 = ["App_ID","Name","Release_Date","Price","About_The_Game","Supported_Languages","Header_Image","Support_URL","Windows","Mac","Linux","Positive","Negative","Achievements","Developers","Publishers","Categories","Genres","Tags","Screenshots","Movies"]

UNIQUE_LANGUAGE = ["english",'French', 'german', 'italian', 'spanish - spain',
       'spanish - latin america', 'simplified chinese',
       'traditional chinese', 'russian', 'japanese', 'korean',
       'portuguese - brazil', 'turkish', 'welsh', 'vietnamese', 'danish',
       'portuguese - portugal', 'dutch', 'polish', 'czech', 'ukrainian',
       'arabic', 'bulgarian', 'hungarian', 'greek', 'norwegian',
       'romanian', 'thai', 'finnish', 'swedish', 'croatian', 'estonian',
       'hebrew', 'icelandic', 'latvian', 'lithuanian', 'maori', 'slovak',
       'slovenian', 'indonesian', 'serbian', 'uzbek', 'urdu', 'armenian',
       'igbo', 'sindhi', 'sinhala', 'cherokee', 'galician', 'catalan',
       'afrikaans', 'kannada', 'luxembourgish', 'hindi', 'gujarati',
       'kyrgyz', 'kazakh', 'turkmen', 'kinyarwanda',
       'tajik', 'odia', 'konkani', 'bangla', 'nepali', 'basque',
       'tigrinya', 'swahili', 'punjabi (gurmukhi)', 'punjabi (shahmukhi)',
       'georgian', 'wolof', 'bosnian', 'persian', 'telugu', 'tamil',
       'irish', 'valencian', 'belarusian', 'quechua', 'zulu', 'xhosa',
       'sotho', 'sorani', 'yoruba', 'uyghur', 'scots', 'tswana',
       'filipino', 'mongolian', 'hausa', 'dari', 'azerbaijani', 'amharic',
       'albanian', 'assamese', 'tatar', 'macedonian', 'marathi',
       'malayalam', 'malay', 'maltese', 'khmer', 'german;',
       'hungarian,polish', 'english dutch',
       'traditional chinese (text only)', 'lang_slovakian']

EUROPEAN_LANGUAGES = ["German", "French", "Italian", 'Spanish - Spain', "Portuguese - Portugal"]

TOP_LANGUAGES = 10



days_in_year = 365.25  # moyenne en tenant compte des années bissextiles

months_in_year = 12

MAX_SEQUENCE_LENGTH = 20
