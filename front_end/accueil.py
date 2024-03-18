import streamlit as st
import requests

# Définition des noms des colonnes
text_col_names = ["App_ID", "Release_Date", "Supported_Languages", "Support_URL", "Developers", "Publishers", "Categories", "Genres"]
bool_col_names = ["Windows", "Mac", "Linux"]
num_col_names = ["Achievements","Price"]

# Dictionnaire pour stocker les entrées de l'utilisateur
user_input = {}
st.markdown("<h1 style='text-align: center;'>GameForecast: Prédir les performances de votre jeu à sa sortie</h1>", unsafe_allow_html=True)
st.title("GameForecast: Prédir les performances de votre jeu à sa sortie")
st.write("Saisie des informations du jeu")
# Création des champs de texte dans les trois premières colonnes avec plusieurs lignes
for i, name in enumerate(text_col_names):
    col = st.columns(3)[i % 3]  # Répartir sur trois colonnes
    user_input[name] = col.text_input(name, key=name)

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
if st.button('Prédiction  du rating'):
    # Remplacez 'your_api_endpoint' par l'URL de votre API
    api_endpoint = 'your_api_endpoint'

    # Envoi des données à l'API
    response = requests.post(api_endpoint, json=user_input)

    # Affichage de la réponse de l'API
    st.write(response.text)

if st.button('Prédiction du nombre de joueur à la sortie'):
    # Remplacez 'your_api_endpoint' par l'URL de votre API
    api_endpoint = 'your_api_endpoint'

    # Envoi des données à l'API
    response = requests.post(api_endpoint, json=user_input)

    # Affichage de la réponse de l'API
    st.write(response.text)
