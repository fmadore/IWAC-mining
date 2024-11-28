import asyncio
import aiohttp
import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
import logging
from datetime import datetime
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
import re

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

# Load French language model with progress bar
def download_spacy_model():
    """Download spacy model with progress bar if not already installed."""
    try:
        nlp = spacy.load('fr_dep_news_trf')
        logging.info("Successfully loaded French transformer language model")
        return nlp
    except OSError:
        logging.info("French transformer model not found. Starting download...")
        from spacy.cli import download
        from tqdm import tqdm
        
        class TqdmProgressbar:
            def __init__(self):
                self.pbar = None
            
            def __call__(self, dl_total, dl_count, width=None):
                if self.pbar is None:
                    self.pbar = tqdm(total=dl_total, unit='B', unit_scale=True)
                self.pbar.n = dl_count
                self.pbar.refresh()
                if dl_count == dl_total:
                    self.pbar.close()
                    self.pbar = None
        
        download('fr_dep_news_trf', progress=TqdmProgressbar())
        nlp = spacy.load('fr_dep_news_trf')
        logging.info("Successfully installed and loaded French transformer model")
        return nlp

# Initialize the model
nlp = download_spacy_model()

# Add batch size configuration for transformer
nlp.max_length = 1000000  # Increase max length to handle longer texts

# Load environment variables from .env file
load_dotenv()

OMEKA_BASE_URL = os.getenv('OMEKA_BASE_URL')
OMEKA_KEY_IDENTITY = os.getenv('OMEKA_KEY_IDENTITY')
OMEKA_KEY_CREDENTIAL = os.getenv('OMEKA_KEY_CREDENTIAL')

def preprocess_text(text):
    """
    Preprocess text using the transformer model's advanced capabilities at different levels.
    Returns a dictionary containing processed text at article, paragraph and sentence levels.
    """
    if not text:
        return {
            "article": "",
            "paragraphs": [],
            "sentences": []
        }
    
    # Initial text cleaning
    text = (text.strip()
        .replace('\xa0', ' ')  # Replace non-breaking spaces
        .replace('  ', ' ')    # Replace double spaces
        .replace('«', '"')     # Normalize French quotes
        .replace('»', '"')
        .replace('"', '"')     # Normalize other quotes
        .replace('"', '"')
        .replace(''', "'")     # Normalize apostrophes
        .replace(''', "'")
        .replace('…', '...')   # Normalize ellipsis
        .replace('–', '-')     # Normalize dashes
        .replace('—', '-'))
    
    # Remove multiple spaces (including those created by previous replacements)
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Process with spaCy
    doc = nlp(text)
    
    def process_tokens(tokens):
        """Helper function to process a sequence of tokens."""
        return [
            token.lemma_
            for token in tokens
            if not token.is_stop 
            and not token.is_punct
            and not token.is_space
            and len(token.text.strip()) > 1
            and not token.like_num
            and token.pos_ not in ['SPACE', 'SYM']
        ]
    
    # Process at article level
    article_tokens = process_tokens(doc)
    
    # Process at paragraph level
    paragraphs = []
    for para in text.split('\n\n'):
        if para.strip():
            para_doc = nlp(para)
            para_tokens = process_tokens(para_doc)
            if para_tokens:
                paragraphs.append(" ".join(para_tokens))
    
    # Process at sentence level
    sentences = []
    for sent in doc.sents:
        sent_tokens = process_tokens(sent)
        if sent_tokens:
            sentences.append(" ".join(sent_tokens))
    
    return {
        "article": " ".join(article_tokens),
        "paragraphs": paragraphs,
        "sentences": sentences
    }

semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

async def fetch_data_async(session, url):
    """Fetch data from a given URL with authentication asynchronously."""
    headers = {
        'Authorization': f'ApiKey {OMEKA_KEY_IDENTITY}:{OMEKA_KEY_CREDENTIAL}'
    }
    try:
        async with semaphore:  # Add this line to limit concurrent requests
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        logging.error(f"Failed to fetch data from {url}: {str(e)}")
        return None

async def fetch_ids_from_item_async(session, url):
    """Async version of fetch_ids_from_item."""
    logging.info(f"Starting to fetch IDs from {url}")
    data = await fetch_data_async(session, url)
    
    if not data or '@reverse' not in data or 'dcterms:subject' not in data['@reverse']:
        logging.error(f"Failed to fetch or parse data from {url}")
        return []
    
    potential_urls = [item['@id'] for item in data['@reverse']['dcterms:subject']]
    logging.info(f"Found {len(potential_urls)} potential URLs to process")
    
    # Create progress bar for URL validation
    article_urls = []
    with tqdm(total=len(potential_urls), desc="Validating articles", unit="item") as pbar:
        for url in potential_urls:
            item_data = await fetch_data_async(session, url)
            if not item_data:
                logging.warning(f"Failed to fetch data for URL: {url}")
                pbar.update(1)
                continue
                
            if '@type' in item_data and 'bibo:Article' in item_data['@type']:
                article_urls.append(url)
                title = item_data.get('o:title', 'Untitled')
                logging.info(f"Found article: '{title}' at {url}")
            pbar.update(1)
    
    logging.info(f"Processing complete: Found {len(article_urls)} articles out of {len(potential_urls)} total items")
    return article_urls

def process_article_content(article_data):
    """Process the content of an article and add processed text."""
    if not article_data:
        return article_data
    
    try:
        # Find the content in bibo:content
        if 'bibo:content' in article_data:
            content = article_data['bibo:content']
            if isinstance(content, list):
                for item in content:
                    if '@value' in item:
                        # Add processed versions of the text at different levels
                        item['processed_text'] = preprocess_text(item['@value'])
            elif isinstance(content, dict) and '@value' in content:
                # Add processed versions of the text at different levels
                content['processed_text'] = preprocess_text(content['@value'])
    except Exception as e:
        logging.error(f"Error processing article content: {str(e)}")
    
    return article_data

async def process_urls_async(urls):
    """Process multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        all_data = []
        successful_fetches = 0
        failed_fetches = 0
        
        # Create progress bar for article processing
        pbar = tqdm(total=len(urls), desc="Processing articles", unit="article")
        
        for url in urls:
            data = await fetch_data_async(session, url)
            if data:
                processed_data = process_article_content(data)
                all_data.append(processed_data)
                successful_fetches += 1
            else:
                failed_fetches += 1
            pbar.update(1)
            
        pbar.close()
        return all_data, successful_fetches, failed_fetches

async def main_async():
    item_url = f"{OMEKA_BASE_URL}/items/59"
    
    logging.info("Starting data collection process")
    
    async with aiohttp.ClientSession() as session:
        # Fetch all article URLs
        urls = await fetch_ids_from_item_async(session, item_url)
        if not urls:
            logging.error("No URLs found to process. Exiting.")
            return

        # Process all URLs concurrently
        all_data, successful_fetches, failed_fetches = await process_urls_async(urls)

        # Log summary statistics
        logging.info(f"Data collection completed:")
        logging.info(f"- Total URLs processed: {len(urls)}")
        logging.info(f"- Successful fetches: {successful_fetches}")
        logging.info(f"- Failed fetches: {failed_fetches}")

        # Save all data to a single JSON file
        output_file = os.path.join(script_dir, 'integrisme_data.json')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=4)
            logging.info(f"Data successfully saved to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save data to file: {str(e)}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 