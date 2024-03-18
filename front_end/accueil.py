import streamlit as st
import requests

# Définition des noms des colonnes et leurs étiquettes correspondantes
text_col_names = [("App_ID", "ID de l'application"), ("Supported_Languages", "Langues supportées"), ("Support_URL", "URL du support"), ("Developers", "Développeurs"), ("Publishers", "Éditeurs"), ("Categories", "Catégories")]
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
bool_col_names = ["Windows", "Mac", "Linux"]
num_col_names = [("Achievements", "Nombre de succès disponibles"), ("Price", "Prix en euros")]

# Dictionnaire pour stocker les entrées de l'utilisateur
user_input = {}
all_fields_filled = True  # Indicateur si tous les champs requis sont remplis

st.title("GameForecast: Prédir les performances de votre jeu à sa sortie")
st.write("Saisie des informations concernant le jeu")

# Création des champs de texte
for name, label in text_col_names:
    if name != "Support_URL" and name != "Categories":  # Les champs URL du support et Categories ne sont pas obligatoires
        user_input[name] = st.text_input(label, key=name)
        if user_input[name] == "":  # Vérifie si le champ obligatoire est vide
            all_fields_filled = False
    elif name == "Support_URL":  # Gestion spéciale pour URL du support
        user_input[name] = st.text_input(label, key=name, value="Aucune")
        if user_input[name] == "Aucune":  # Assigner None si le champ est vide
            user_input[name] = None

user_input['Release_Date'] = st.date_input("Date de sortie", key='Release_Date')

# Ajout des listes déroulantes pour genres et catégories (ces champs ne sont pas marqués comme obligatoires)
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
    if all_fields_filled:
        api_endpoint = 'your_api_endpoint'
        response = requests.post(api_endpoint, json=user_input)
        st.write(response.text)
    else:
        st.error("Veuillez remplir tous les champs.")

if st.button('Prédiction du nombre de joueurs à la sortie'):
    if all_fields_filled:
        api_endpoint = 'your_api_endpoint'
        response = requests.post(api_endpoint, json=user_input)
        st.write(response.text)
    else:
        st.error("Veuillez remplir tous les champs.")
