import requests
import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
import logging
from datetime import datetime

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set up logging with path in script directory
log_filename = f'integrisme_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_path = os.path.join(script_dir, log_filename)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

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
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch data from {url}: {str(e)}")
        return None

def fetch_ids_from_item(url):
    """Fetch all @id URLs from the @reverse section of the item."""
    logging.info(f"Fetching IDs from {url}")
    data = fetch_data(url)
    
    if data and '@reverse' in data and 'dcterms:subject' in data['@reverse']:
        urls = [item['@id'] for item in data['@reverse']['dcterms:subject']]
        logging.info(f"Found {len(urls)} URLs to process")
        return urls
    else:
        logging.warning(f"No @id URLs found in {url}")
        return []

def main():
    # URL of the item to fetch @id URLs from
    item_url = f"{OMEKA_BASE_URL}/items/59"
    
    logging.info("Starting data collection process")
    
    # Fetch all @id URLs
    urls = fetch_ids_from_item(item_url)
    if not urls:
        logging.error("No URLs found to process. Exiting.")
        return

    # List to store all fetched data
    all_data = []
    successful_fetches = 0
    failed_fetches = 0

    # Fetch data for each URL with progress bar
    for url in tqdm(urls, desc="Fetching articles", unit="article"):
        data = fetch_data(url)
        if data:
            all_data.append(data)
            successful_fetches += 1
        else:
            failed_fetches += 1

    # Log summary statistics
    logging.info(f"Data collection completed:")
    logging.info(f"- Total URLs processed: {len(urls)}")
    logging.info(f"- Successful fetches: {successful_fetches}")
    logging.info(f"- Failed fetches: {failed_fetches}")

    # Save all data to a single JSON file in script directory
    output_file = os.path.join(script_dir, 'integrisme_data.json')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Data successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save data to file: {str(e)}")

if __name__ == "__main__":
    main() 