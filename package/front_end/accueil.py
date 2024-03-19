import streamlit as st
import requests
#from package.scripts.params import *
#from package.scripts.utils import verify_required_fields

def verify_required_fields(user_inputs, required_fields_list):
    """Vérifie si tous les champs requis sont remplis."""
    for field in required_fields_list:
        if not user_inputs.get(field):  # Cela vérifie si le champ est vide, zéro, etc.
            return False
    return True


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








# Définition des noms des colonnes et leurs étiquettes correspondantes
text_col_names = [("App_ID", "ID du jeu"), ("Developers", "Développeurs"), ("Publishers", "Éditeurs")]
bool_col_names = ["Windows", "Mac", "Linux"]
num_col_names = [("Achievements", "Nombre de succès disponibles"), ("Price", "Prix en euros")]

# Dictionnaire pour stocker les entrées de l'utilisateur
user_input = {}

st.title("👾 GameForecast: Prédir les performances de votre jeu à sa sortie 👾")
st.write("🕹️ Saisie des informations concernant le jeu 🕹️")

# Création des champs de texte
col1, col2, col3 = st.columns(3)

# Assign the text to each column
with col1:
    user_input[text_col_names[0][0]] = st.text_input(text_col_names[0][1], key=text_col_names[0][0])

with col2:
    user_input[text_col_names[1][0]] = st.text_input(text_col_names[1][1], key=text_col_names[1][0])

with col3:
    user_input[text_col_names[2][0]] = st.text_input(text_col_names[2][1], key=text_col_names[2][0])


#autres paramètres
user_input["Support_URL"] = st.text_input("URL du support", key="Support_URL", value="Aucune")
if user_input["Support_URL"] == "Aucune":  # Assigner None si le champ est vide
    user_input["Support_URL"] = None

user_input['Release_Date'] = st.date_input("Date de sortie", key='Release_Date')

# Ajout des listes déroulantes pour genres et catégories (ces champs ne sont pas marqués comme obligatoires)
user_input['Supported_Languages'] = st.multiselect('Langues disponibles', languages_options, key='Supported_Languages')
user_input['Genres'] = st.multiselect('Genres', genre_options, key='Genres')
user_input['Categories'] = st.multiselect('Catégories', category_options, key='Categories')

# Création des champs booléens et numériques
st.write("OS supportés :")
bool_cols = st.columns(len(bool_col_names))
for col, name in zip(bool_cols, bool_col_names):
    user_input[name] = col.checkbox(name, key=name)

num_cols = st.columns(len(num_col_names))
for col, (name, label) in zip(num_cols, num_col_names):
    user_input[name] = col.number_input(label, min_value=0.0 if name == "Price" else 0, value=0.0 if name == "Price" else 0, step=0.01 if name == "Price" else 1, key=name)

# Bouton pour envoyer les données
if st.button('Prédiction du rating'):
    all_fields_filled = verify_required_fields(user_input, required_fields)
    if all_fields_filled:
        api_endpoint = 'your_api_endpoint'
        response = requests.post(api_endpoint, json=user_input)
        if response.ok:
            st.write(response.text)
        else:
            st.error("Une erreur s'est produite avec l'API.")
    else:
        st.error("Veuillez remplir tous les champs.")

if st.button('Prédiction du nombre de joueurs à la sortie'):
    all_fields_filled = verify_required_fields(user_input, required_fields)
    if all_fields_filled:
        api_endpoint = 'your_api_endpoint'
        response = requests.post(api_endpoint, json=user_input)
        if response.ok:
            st.write(response.text)
        else:
            st.error("Une erreur s'est produite avec l'API.")
    else:
        st.error("Veuillez remplir tous les champs.")
