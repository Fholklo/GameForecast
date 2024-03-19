import os

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
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
