import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras
from package.scripts.params import LOCAL_REGISTRY_PATH,MODEL_TARGET,folder_path


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

def save_model(model_name:str="model_rating",model_type : str=None, model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"

    """

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
    model_type (str): Type de modèle ('model_num', 'model_text', 'model_image').

    Returns:
    Le modèle chargé si trouvé, sinon None.
    """
    # Construire le chemin complet vers le sous-dossier du modèle
    file_folder_model = os.path.join(folder_path, model_name, model_type)

    # Vérifier si le dossier existe
    if os.path.exists(file_folder_model):
        # Liste de tous les fichiers de modèle dans le sous-dossier
        model_files = sorted([os.path.join(file_folder_model, f) for f in os.listdir(file_folder_model) if os.path.isfile(os.path.join(file_folder_model, f))])
        # Trouver le chemin du modèle le plus récent, si la liste n'est pas vide
        most_recent_model_path = model_files[-1] if model_files else None
    else:
        most_recent_model_path = None

    # Charger et retourner le modèle le plus récent, si existant
    if most_recent_model_path:
        print(f"✅ {model_type} loaded ")
        return keras.models.load_model(most_recent_model_path)

    else:
        print(f"Aucun modèle trouvé dans le dossier spécifié pour {model_type}.")
        return None
