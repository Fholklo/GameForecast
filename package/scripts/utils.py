def verify_required_fields(user_inputs, required_fields_list):
    """Vérifie si tous les champs requis sont remplis."""
    for field in required_fields_list:
        if not user_inputs.get(field):  # Cela vérifie si le champ est vide, zéro, etc.
            return False
    return True
