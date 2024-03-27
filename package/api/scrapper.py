import requests
import json
import re

DEFAULT_TIMEOUT = 10
DEFAULT_CURRENCY = 'EUR'  # Changed to 'EUR' as 'fr' is not a currency code
DEFAULT_LANGUAGE = 'english'  # Changed to 'french' as 'fr' might be misinterpreted

def sanitize_text(text):
    '''
    Removes HTML codes and reduces spaces.
    '''
    text = re.sub('<[^<]+?>', ' ', text)# Remove HTML tags
    text = text.replace('*', ' ')
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    clean_text = ', '.join([lang.strip() for lang in text.split(',')])  # Divise le texte par les virgules et reconstruit avec une structure uniforme.
    return clean_text

def price_to_float(price_str):
    '''
    Converts a price string to a float value, handling different formats.
    '''
    price_str = price_str.replace(',', '.')
    matches = re.findall(r'(\d+(?:\.\d+)?)', price_str)
    return float(matches[0]) if matches else 0.0

def do_request(url, parameters=None):
    '''
    Performs a web request and returns the JSON response.
    '''
    try:
        response = requests.get(url, params=parameters, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred.
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'Request error: {e}')
        return None

def parse_steam_game(app_data):
    '''
    Extracts specific fields from the Steam game data.
    '''
    if not app_data.get('success', False):
        return None

    app_data = app_data['data']
    game_info = {
        'Name': app_data.get('name', ''),
        'Supported_Languages': sanitize_text(app_data.get('supported_languages', '')),
        'Support_URL': app_data.get('support_info', {}).get('url', ''),
        'Developers': ', '.join(app_data.get('developers', [])),
        'Publishers': ', '.join(app_data.get('publishers', [])),
        'Release_Date': app_data.get('release_date', {}).get('date', ''),
        'Genres': ', '.join([genre['description'] for genre in app_data.get('genres', [])]),
        'Categories': ', '.join([category['description'] for category in app_data.get('categories', [])]),
        'Windows': app_data.get('platforms', {}).get('windows', False),
        'Mac': app_data.get('platforms', {}).get('mac', False),
        'Linux': app_data.get('platforms', {}).get('linux', False),
        'Achievements': app_data.get('achievements', {}).get('total', 0),
        'Price': price_to_float(app_data.get('price_overview', {}).get('final_formatted', '0')) if 'price_overview' in app_data else 0,
        'About_The_Game': sanitize_text(app_data.get('about_the_game', '')),
        'Screenshots': [screenshot['path_full'] for screenshot in app_data.get('screenshots', [])],
        'Header_Image': app_data.get('header_image', '')
    }
    return game_info

def get_steam_game_info(app_id, currency=DEFAULT_CURRENCY, language=DEFAULT_LANGUAGE):
    '''
    Fetches and processes information for a single Steam app ID.
    '''
    url = "https://store.steampowered.com/api/appdetails"
    params = {"appids": app_id, "cc": currency, "l": language}
    response = do_request(url, params)
    if response and str(app_id) in response:
        return parse_steam_game(response[str(app_id)])
    else:
        return None