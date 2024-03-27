
import os
import time
import pickle


from tensorflow import keras
from package.scripts.params import folder_path


def save_results(history: dict, model_name: str, model_type: str) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save history locally
    if history is not None:
        history_path = os.path.join(folder_path, model_name, model_type, f"{model_name}_{model_type}_{timestamp}.pkl")
        with open(history_path, "wb") as file:
            pickle.dump(history, file)

    print("✅ Results saved locally")

def save_model(model_name:str="model_rating",model_type : str=None, model: keras.Model = None) -> None:


    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(folder_path,model_name,model_type, f"{model_name}_{model_type}_{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    return None

def load_most_recent_model(folder_path, model_name, model_type):
    """
    Charge le modèle le plus récent à partir d'un sous-dossier spécifié.

    Args:
    folder_path (str): Chemin du dossier contenant les sous-dossiers des modèles.
    model_name (str): Nom du modèle.
    model_type (str): Type de modèle ('model_num', 'model_text', 'model_image').

    Returns:
    Le modèle chargé si trouvé, sinon None.
    """
    # Construire le chemin complet vers le sous-dossier du modèle
    file_folder_model = os.path.join(folder_path, model_name, model_type)

    # Vérifier si le dossier existe
    if os.path.exists(file_folder_model):
        # Liste de tous les fichiers de modèle .h5 dans le sous-dossier
        model_files = sorted([os.path.join(file_folder_model, f) for f in os.listdir(file_folder_model) if os.path.isfile(os.path.join(file_folder_model, f)) and f.endswith('.h5')])
        # Trouver le chemin du modèle le plus récent, si la liste n'est pas vide
        most_recent_model_path = model_files[-1] if model_files else None
    else:
        most_recent_model_path = None

    # Charger et retourner le modèle le plus récent, si existant
    if most_recent_model_path:
        print(f"✅ Chargement réussi du modèle {model_type} à partir de {most_recent_model_path}")
        return keras.models.load_model(most_recent_model_path)
    else:
        print(f"Aucun modèle trouvé dans le dossier spécifié pour {model_type}.")
        return None
