import streamlit as st
import requests
from params_acc import SERVICE_URL

# Définition des noms des colonnes et leurs étiquettes correspondantes
background_image_url ="https://www.la-console-retro.fr/cdn/shop/articles/actualite-lexplosion-du-marche-du-jeux-video-retrogaming-558644.jpg"

background_css = f"""
<style>
.stApp {{
    background-image: url("{background_image_url}");
    background-size: cover;
    background-position: center;
}}
</style>
"""

# Injection du CSS dans la page
st.markdown(background_css, unsafe_allow_html=True)

entries_css = f"""
<style>
.dataEntry {{
    background-color: rgba(0, 0, 0, 0.8); /* Noir avec 80% d'opacité */
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}}
.titleBox {{
    background-color: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: center; /* Centre le texte */
    display: flex; /* Utilise Flexbox */
    justify-content: center; /* Centre les enfants horizontalement dans Flexbox */
    align-items: center; /* Centre les enfants verticalement dans Flexbox */
    height: 100px; /* Ou toute autre hauteur, ajustez selon vos besoins */
}}
.centeredText {{
    text-align: center; /* Centre le texte horizontalement */
}}
.customError {{
    background-color: rgba(255, 0, 0, 0.8); /* Rouge avec 80% d'opacité */
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}}
</style>
"""

# Injection du CSS dans la page
st.markdown(entries_css, unsafe_allow_html=True)


# Dictionnaire pour stocker les entrées de l'utilisateur
st.markdown(f"""<div class='dataEntry centeredText'><h1>👾 GameForecast:
            Prédir les performances de votre jeu à sa sortie 👾</h1></div>""", unsafe_allow_html=True)

st.markdown(f"<div class='dataEntry centeredText'><h2>🕹️ Saisir l'ID Steam du jeu 🕹️</h2></div>", unsafe_allow_html=True)
game_id = st.text_input('ID du jeu', label_visibility='collapsed')

# Bouton pour envoyer les données
if st.button('Prédir le rating du jeu et son nombre de joueur'):
    with st.spinner('Chargement en cours...'):
        api_endpoint = SERVICE_URL+"/App_ID"
        response = requests.get(api_endpoint,  params={'App_ID': game_id})
        if response.ok:
            data = response.json()
            name = data["Name"]
            rating = data["Prediction_rating"]
            player = data["Prediction_player"]
            url_header = data["Header_Image"]
            url_screenshot = data["Screenshots"]
            support = "Oui" if data['Support_URL'] else "Non"
            st.markdown(f"<div class='titleBox'><h1>{name}</h1></div>", unsafe_allow_html=True)

            # HTML pour centrer l'image
            html_string = f"""
            <div style="display:flex;justify-content:center; margin-bottom:10px;">
                <img src="{url_header}" style="max-width:60%;height:auto;">
            </div>
            """
            # Utiliser st.markdown pour afficher l'image centrée
            st.markdown(html_string, unsafe_allow_html=True)

            st.markdown(f"<div class='dataEntry'><h6 style='font-weight:bold;'>{name} aura un rating de {rating} et environ {player} joueurs</h6></div>", unsafe_allow_html=True)

            # Utilisez également la classe dataEntry pour l'introduction aux données
            st.markdown(f"<div class='dataEntry'>Voici les données paramètres récoltées et utilisées pour la prédiction :</div>", unsafe_allow_html=True)

            # Modifiez les appels à st.markdown pour utiliser la classe .dataEntry
            st.markdown(f"<div class='dataEntry'><span style='text-decoration: underline;'>Déscription</span> : {data.get('About_The_Game', 'Information non disponible')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dataEntry'><span style='text-decoration: underline;'>Développeurs</span> : {data.get('Developers', 'Information non disponible')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dataEntry'><span style='text-decoration: underline;'>Editeurs</span> : {data.get('Publishers', 'Information non disponible')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dataEntry'><span style='text-decoration: underline;'>Genres</span> : {data.get('Genres', 'Information non disponible')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dataEntry'><span style='text-decoration: underline;'>Catégories</span> : {data.get('Categories', 'Information non disponible')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dataEntry'><span style='text-decoration: underline;'>Prix</span> : {data.get('Price', 'Information non disponible')} euros</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dataEntry'><span style='text-decoration: underline;'>Langues disponibles</span> : {data.get('Supported_Languages', 'Information non disponible')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dataEntry'><span style='text-decoration: underline;'>SAV</span> : {support}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dataEntry'><span style='text-decoration: underline;'>Nombre de succès sur Steam</span> : {data.get('Achievements', 'Information non disponible')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='dataEntry'><span style='text-decoration: underline;'>Screenshot utilisé pour le CNN</span> : </div>", unsafe_allow_html=True)
            st.image(url_screenshot)
            st.balloons()
        else:
            st.markdown('<div class="customError">L\'ID soumise n\'est pas reconnue ou les données du jeu ne sont pas encore accessibles sur steam</div>', unsafe_allow_html=True)
