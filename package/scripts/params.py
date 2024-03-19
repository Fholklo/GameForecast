import os

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")

PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")

GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")

############## CONSTANTS ###################
#Path
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "gameforecast", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "gameforecast", "training_outputs")



#Data
genre_options = ['Action', 'Casual', 'Indie', 'RPG', 'Simulation', 'Adventure',
                 'Strategy', 'Design & Illustration', 'Video Production',
                 'Early Access', 'Massively Multiplayer', 'Free to Play', 'Sports',
                 'Animation & Modeling', 'Utilities', 'Game Development',
                 'Photo Editing', 'Software Training', 'Nudity', 'Violent',
                 'Racing', 'Gore', 'Sexual Content', 'Audio Production',
                 'Web Publishing', 'Movie', 'Education', 'Accounting']
category_options = ['Single-player', 'Steam Cloud', 'Family Sharing', 'Steam Achievements',
                    'Partial Controller Support', 'Full controller support', 'Multi-player',
                    'Steam Trading Cards', 'Steam Workshop', 'Co-op', 'Online Co-op',
                    'Steam Leaderboards', 'PvP', 'Online PvP', 'Remote Play on Phone',
                    'Remote Play on Tablet', 'Remote Play on TV', 'In-App Purchases',
                    'Tracked Controller Support', 'VR Only', 'MMO', 'Cross-Platform Multiplayer',
                    'Stats', 'Includes level editor', 'Shared/Split Screen',
                    'Remote Play Together', 'No', 'VR Supported', 'Captions available',
                    'VR Support', 'Shared/Split Screen PvP', 'Shared/Split Screen Co-op',
                    'Valve Anti-Cheat enabled', 'LAN Co-op', 'Steam Turn Notifications',
                    'HDR available', 'LAN PvP', 'Commentary available', 'Includes Source SDK',
                    'SteamVR Collectibles', 'Mods', 'Mods (require HL2)']

languages_options = ["German", "French", "Italian", 'Spanish - Spain', "Portuguese - Portugal", 'English',
                     'Simplified Chinese', 'Russian', 'Japanese', 'Korean', 'Traditional Chinese',
                     'Portuguese - Brazil', 'Polish', 'Turkish']

required_fields = ['App_ID', 'Developers', 'Publishers', 'Achievements', 'Price']
