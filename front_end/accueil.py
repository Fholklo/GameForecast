import streamlit as st
import requests

# Définition des noms des colonnes
text_col_names = ["App_ID", "Release_Date", "Supported_Languages", "Support_URL", "Developers", "Publishers"]
category_options = ['Single-player', 'Steam Cloud', 'Family Sharing', 'Steam Achievements', 'Partial Controller Support', 'Full controller support', 'Multi-player', 'Steam Trading Cards', 'Steam Workshop', 'Co-op', 'Online Co-op', 'Steam Leaderboards', 'PvP', 'Online PvP', 'Remote Play on Phone', 'Remote Play on Tablet', 'Remote Play on TV', 'In-App Purchases', 'Tracked Controller Support', 'VR Only', 'MMO', 'Cross-Platform Multiplayer', 'Stats', 'Includes level editor', 'Shared/Split Screen', 'Remote Play Together', 'No', 'VR Supported', 'Captions available', 'VR Support', 'Shared/Split Screen PvP', 'Shared/Split Screen Co-op', 'Valve Anti-Cheat enabled', 'LAN Co-op', 'Steam Turn Notifications', 'HDR available', 'LAN PvP', 'Commentary available', 'Includes Source SDK', 'SteamVR Collectibles', 'Mods', 'Mods (require HL2)']
genre_options = ['Action', 'Casual', 'Indie', 'RPG', 'Simulation', 'Adventure', 'Strategy', 'Design & Illustration', 'Video Production', 'Early Access', 'Massively Multiplayer', 'Free to Play', 'Sports', 'Animation & Modeling', 'Utilities', 'Game Development', 'Photo Editing', 'Software Training', 'Nudity', 'Violent', 'Racing', 'Gore', 'Sexual Content', 'Audio Production', 'Web Publishing', 'Movie', 'Education', 'Accounting']
bool_col_names = ["Windows", "Mac", "Linux"]
num_col_names = ["Achievements", "Price"]

# Dictionnaire pour stocker les entrées de l'utilisateur
user_input = {}

st.title("GameForecast: Prédir les performances de votre jeu à sa sortie")
st.write("Saisie des informations du jeu")

# Création des champs de texte dans les trois premières colonnes avec plusieurs lignes
for i in range(0, len(text_col_names), 3):  # Itérer par pas de 3
    cols = st.columns(3)  # Créer trois colonnes
    for j in range(3):
        if i + j < len(text_col_names):  # Vérifier si le champ existe
            name = text_col_names[i + j]
            user_input[name] = cols[j].text_input(name, key=name)

# Ajout de la liste déroulante pour le genre
user_input['Genres'] = st.multiselect('Genres test', genre_options, key='Genres')

# Ajout de la liste déroulante pour les catégories
user_input['Categories'] = st.multiselect('Categories', category_options, key='Categories')

# Création d'une nouvelle ligne pour les champs booléens
st.write("OS supportés :")  # Titre optionnel pour la section
bool_cols = st.columns(len(bool_col_names))
for col, name in zip(bool_cols, bool_col_names):
    user_input[name] = col.checkbox(name, key=name)

# Création d'une nouvelle ligne pour les champs numériques
st.write("Numériques :")  # Titre optionnel pour la section
num_cols = st.columns(len(num_col_names))
for col, name in zip(num_cols, num_col_names):
    user_input[name] = col.number_input(name, min_value=0, value=0, step=1, key=name)

# Bouton pour envoyer les données
if st.button('Prédiction du rating'):
    # Remplacez 'your_api_endpoint' par l'URL de votre API
    api_endpoint = 'your_api_endpoint'

    # Envoi des données à l'API
    response = requests.post(api_endpoint, json=user_input)

    # Affichage de la réponse de l'API
    st.write(response.text)

if st.button('Prédiction du nombre de joueurs à la sortie'):
    # Remplacez 'your_api_endpoint' par l'URL de votre API
    api_endpoint = 'your_api_endpoint'

    # Envoi des données à l'API
    response = requests.post(api_endpoint, json=user_input)

    # Affichage de la réponse de l'API
    st.write(response.text)
