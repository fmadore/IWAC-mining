import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

OMEKA_BASE_URL = os.getenv('OMEKA_BASE_URL')
OMEKA_KEY_IDENTITY = os.getenv('OMEKA_KEY_IDENTITY')
OMEKA_KEY_CREDENTIAL = os.getenv('OMEKA_KEY_CREDENTIAL')

def fetch_data(url):
    """Fetch data from a given URL with authentication."""
    headers = {
        'Authorization': f'ApiKey {OMEKA_KEY_IDENTITY}:{OMEKA_KEY_CREDENTIAL}'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data from {url}")
        return None

def fetch_ids_from_item(url):
    """Fetch all @id URLs from the @reverse section of the item."""
    data = fetch_data(url)
    if data and '@reverse' in data and 'dcterms:subject' in data['@reverse']:
        return [item['@id'] for item in data['@reverse']['dcterms:subject']]
    else:
        print(f"No @id URLs found in {url}")
        return []

def main():
    # URL of the item to fetch @id URLs from
    item_url = f"{OMEKA_BASE_URL}/items/59"
    
    # Fetch all @id URLs
    urls = fetch_ids_from_item(item_url)

    # List to store all fetched data
    all_data = []

    # Fetch data for each URL
    for url in urls:
        data = fetch_data(url)
        if data:
            all_data.append(data)

    # Save all data to a single JSON file
    with open('integrisme_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print("Data has been saved to integrisme_data.json")

if __name__ == "__main__":
    main() 