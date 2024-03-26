import streamlit as st
import requests
from params_acc import languages_options,genre_options,category_options,SERVICE_URL

# Définition des noms des colonnes et leurs étiquettes correspondantes
text_col_names = [("Developers", "Développeurs"), ("Publishers", "Éditeurs")]
bool_col_names = ["Windows", "Mac", "Linux"]
num_col_names = [("Achievements", "Nombre de succès disponibles"), ("Price", "Prix en euros")]

# Dictionnaire pour stocker les entrées de l'utilisateur
user_input = {}

st.title("👾 GameForecast: Prédir les performances de votre jeu à sa sortie 👾")
st.write("🕹️ Saisie des informations concernant le jeu 🕹️")

# Création des champs de texte
col2, col3 = st.columns(2)

with col2:
    user_input[text_col_names[0][0]] = st.text_input(text_col_names[0][1], key=text_col_names[0][0])

with col3:
    user_input[text_col_names[1][0]] = st.text_input(text_col_names[1][1], key=text_col_names[1][0])


#autres paramètres
user_input["Support_URL"] = st.text_input("URL du support", key="Support_URL", value="Aucune")
if user_input["Support_URL"] == "Aucune":  # Assigner None si le champ est vide
    user_input["Support_URL"] = None

user_input['Release_Date'] = st.date_input("Date de sortie", key='Release_Date')
user_input['Release_Date'] = user_input['Release_Date'].isoformat()
# Ajout des listes déroulantes pour genres et catégories (ces champs ne sont pas marqués comme obligatoires)

user_input['Supported_Languages'] = st.multiselect('Langues disponibles', languages_options, key='Supported_Languages')
languages_list = ",".join(user_input["Supported_Languages"])
user_input['Supported_Languages'] = languages_list

user_input['Genres'] = st.multiselect('Genres', genre_options, key='Genres')
languages_list = ",".join(user_input["Genres"])
user_input['Genres'] = languages_list

user_input['Categories'] = st.multiselect('Catégories', category_options, key='Categories')
languages_list = ",".join(user_input["Categories"])
user_input['Categories'] = languages_list

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
    api_endpoint = SERVICE_URL+"/predict_rating"
    response = requests.get(api_endpoint, params=user_input)
    st.write(response)
    if response.ok:
        st.write(response.text)
    else:
        st.error("Une erreur s'est produite avec l'API.")
else:
    st.error("Veuillez remplir tous les champs.")

if st.button('Prédiction du nombre de joueurs à la sortie'):
    api_endpoint = SERVICE_URL
    response = requests.post(api_endpoint, json=user_input)
    if response.ok:
        st.write(response.text)
    else:
        st.error("Une erreur s'est produite avec l'API.")
else:
    st.error("Veuillez remplir tous les champs.")
