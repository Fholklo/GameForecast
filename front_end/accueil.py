import streamlit as st
import requests

# Définition des noms des colonnes
col_names = [
    "App_ID", "Release_Date", "Price", "Supported_Languages", "Support_URL",
    "Windows", "Mac", "Linux", "Achievements", "Developers",
    "Publishers", "Categories", "Genres"
]

# Création des colonnes dans Streamlit
cols = st.columns(13)

# Dictionnaire pour stocker les entrées de l'utilisateur
user_input = {}

# Boucle pour générer des champs de saisie dans chaque colonne
for i, col in enumerate(cols):
    # Utilisation des checkboxes pour les booléens (Windows, Mac, Linux).
    if col_names[i] in ["Windows", "Mac", "Linux"]:
        user_input[col_names[i]] = col.checkbox(col_names[i], key=col_names[i])
    # Utilisation d'un champ numérique pour les Achievements.
    elif col_names[i] == "Achievements":
        user_input[col_names[i]] = col.number_input(col_names[i], min_value=0, value=0, step=1, key=col_names[i])
    else:
        # Utilisation de champs de texte pour les autres types de données.
        user_input[col_names[i]] = col.text_input(col_names[i], key=col_names[i])

# Bouton pour envoyer les données
if st.button('Envoyer les données'):
    # Remplacez 'your_api_endpoint' par l'URL de votre API
    api_endpoint = 'your_api_endpoint'

    # Envoi des données à l'API
    response = requests.post(api_endpoint, json=user_input)

    # Affichage de la réponse de l'API
    st.write(response.text)
