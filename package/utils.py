import os

def remove_h5_and_pkl_files(folder_path="package/tok_preproc_model"):
    # Parcourir tous les fichiers dans le dossier et les sous-dossiers
    for root_dir, dirs, files in os.walk(folder_path):
        for file in files:
            # Construire le chemin complet du fichier
            file_path = os.path.join(root_dir, file)

            # Supprimer les fichiers .h5 dans tous les dossiers
            if file.endswith('.h5'):
                os.remove(file_path)
                print(f"Supprimé: {file_path}")

            # Supprimer les fichiers .pkl uniquement dans les sous-dossiers (pas dans le dossier racine)
            if root_dir != folder_path and file.endswith('.pkl'):
                os.remove(file_path)
                print(f"Supprimé: {file_path}")


remove_h5_and_pkl_files()
