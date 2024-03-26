import os

def remove_h5_files(folder_path="package/tok_preproc_model"):
    # Parcourir tous les fichiers dans le dossier et les sous-dossiers
    for root_dir, dirs, files in os.walk(folder_path):
        for file in files:
            # Vérifier si le fichier se termine par .h5
            if file.endswith('.h5'):
                # Construire le chemin complet du fichier
                file_path = os.path.join(root_dir, file)
                # Supprimer le fichier
                os.remove(file_path)
                # Afficher le chemin du fichier supprimé
                print(f"Supprimé: {file_path}")

remove_h5_files()
