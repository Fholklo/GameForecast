#Data
genre_options = ['Action', 'Casual', 'Indie', 'RPG', 'Simulation', 'Adventure',
                 'Strategy', 'Design & Illustration', 'Video Production',
                 'Early Access', 'Massively Multiplayer', 'Free to Play', 'Sports',
                 'Animation & Modeling', 'Utilities', 'Game Development',
                 'Photo Editing', 'Software Training', 'Nudity', 'Violent',
                 'Racing', 'Gore', 'Sexual Content', 'Audio Production',
                 'Web Publishing', 'Movie', 'Education', 'Accounting']
category_options = ['Single-player', 'Steam Cloud', 'Family Sharing', 'Steam Achievements',
                    'Partial Controller Support', 'Full controller support', 'Multi-player',
                    'Steam Trading Cards', 'Steam Workshop', 'Co-op', 'Online Co-op',
                    'Steam Leaderboards', 'PvP', 'Online PvP', 'Remote Play on Phone',
                    'Remote Play on Tablet', 'Remote Play on TV', 'In-App Purchases',
                    'Tracked Controller Support', 'VR Only', 'MMO', 'Cross-Platform Multiplayer',
                    'Stats', 'Includes level editor', 'Shared/Split Screen',
                    'Remote Play Together', 'No', 'VR Supported', 'Captions available',
                    'VR Support', 'Shared/Split Screen PvP', 'Shared/Split Screen Co-op',
                    'Valve Anti-Cheat enabled', 'LAN Co-op', 'Steam Turn Notifications',
                    'HDR available', 'LAN PvP', 'Commentary available', 'Includes Source SDK',
                    'SteamVR Collectibles', 'Mods', 'Mods (require HL2)']

languages_options = ["German", "French", "Italian", 'Spanish - Spain', "Portuguese - Portugal", 'English',
                     'Simplified Chinese', 'Russian', 'Japanese', 'Korean', 'Traditional Chinese',
                     'Portuguese - Brazil', 'Polish', 'Turkish']

required_fields = ['App_ID', 'Developers', 'Publishers', 'Achievements', 'Price']

SERVICE_URL="https://gameforecast-fquujalnla-ew.a.run.app"

#fonction

def verify_required_fields(user_inputs, required_fields_list):
    """Vérifie si tous les champs requis sont remplis."""
    for field in required_fields_list:
        if not user_inputs.get(field):  # Cela vérifie si le champ est vide, zéro, etc.
            return False
    return True
