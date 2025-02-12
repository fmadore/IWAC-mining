"""
This module downloads and processes article data from any Omeka item.
It handles authentication, concurrent data fetching, and text preprocessing using spaCy.

Main Features:
-------------
1. Authentication and API Access:
   - Uses environment variables for secure API authentication
   - Handles concurrent API requests with rate limiting
   
2. Data Collection:
   - Fetches articles based on user-provided Omeka item ID
   - Validates article types and metadata
   - Provides progress tracking for long-running operations

3. Text Processing:
   - Uses spaCy's French transformer model for advanced NLP
   - Normalizes text (quotes, spaces, dashes)
   - Handles French-specific text features (contractions, accents)
   - Processes text at multiple levels (article, paragraph, sentence)

4. Error Handling and Logging:
   - Comprehensive logging system with file and console output
   - Graceful error handling for network and processing issues
   - Detailed progress tracking with tqdm progress bars

Requirements:
------------
- Python 3.7+
- spaCy with French transformer model (fr_dep_news_trf)
- Environment variables:
  - OMEKA_BASE_URL: Base URL for the Omeka API
  - OMEKA_KEY_IDENTITY: API key identity
  - OMEKA_KEY_CREDENTIAL: API key credential

Output:
-------
- Processed data saved as JSON file (named based on item title)
- Detailed logs in the logs directory
- Progress information in the console

Usage:
------
1. Ensure environment variables are set in .env file
2. Run the script: python omeka_item_downloader.py
3. Enter the Omeka item ID when prompted
4. Monitor progress in console and logs
5. Access processed data in the generated JSON file (named after the item's title)
"""

# Initialize welcome message before any imports or processing
print("\n=== Omeka Article Downloader ===")
print("Initializing system...")
print("Loading French language model (this may take a few moments)...")

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
import warnings

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add warning filter for PyTorch future warning
warnings.filterwarnings('ignore', category=FutureWarning, module='thinc.shims.pytorch')

def setup_logging(item_id=None):
    """
    Set up logging configuration with a well-structured filename.
    
    Args:
        item_id (str, optional): Omeka item ID to include in the log filename
        
    Returns:
        str: Path to the created log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if item_id:
        log_filename = f'omeka_download_item{item_id}_{timestamp}.log'
    else:
        log_filename = f'omeka_download_{timestamp}.log'
    
    log_path = os.path.join(logs_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Log file created at: {log_path}")
    return log_path

def download_spacy_model():
    """
    Download and load the French spaCy transformer model.
    
    This function attempts to load the French transformer model and downloads it
    if not already installed. It includes a custom progress bar for the download process.
    
    Note:
        Suppresses PyTorch FutureWarning about model loading as we're using official spaCy models.
        This is safe as we're only loading verified models from spaCy's model repository.
    
    Returns:
        spacy.language.Language: Loaded spaCy language model
    
    Raises:
        OSError: If model download fails
    """
    from tqdm import tqdm
    
    try:
        # Safe to use default loading as we're using official spaCy models
        with tqdm(total=1, desc="Loading spaCy model", unit="model") as pbar:
            nlp = spacy.load('fr_dep_news_trf')
            pbar.update(1)
        logging.info("Successfully loaded French transformer language model")
        return nlp
    except OSError:
        logging.info("French transformer model not found. Starting download...")
        from spacy.cli import download
        
        class TqdmProgressbar:
            def __init__(self):
                self.pbar = None
            
            def __call__(self, dl_total, dl_count, width=None):
                if self.pbar is None:
                    self.pbar = tqdm(total=dl_total, unit='B', unit_scale=True, desc="Downloading model")
                self.pbar.n = dl_count
                self.pbar.refresh()
                if dl_count == dl_total:
                    self.pbar.close()
                    self.pbar = None
        
        download('fr_dep_news_trf', progress=TqdmProgressbar())
        print("\nLoading downloaded model...")
        with tqdm(total=1, desc="Loading spaCy model", unit="model") as pbar:
            nlp = spacy.load('fr_dep_news_trf')
            pbar.update(1)
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
    Preprocess French text using advanced NLP techniques and spaCy's transformer model.
    
    Text Processing Strategy:
    -----------------------
    1. Character Normalization:
       - Standardizes quotes, spaces, and dashes
       - Preserves case to maintain proper nouns and sentence structure
       - Handles French-specific characters (accents, cedillas)
    
    2. French Contractions:
       - Preserves common contractions (l', d', qu', n', etc.)
       - Maintains elision forms to respect French grammar
       - Handles special cases like "l'on", "d'accord", "qu'il"
    
    3. Token Processing:
       - Uses spaCy's French transformer for accurate tokenization
       - Preserves case in lemmatization for proper nouns
       - Removes stopwords while keeping important French particles
    
    Rationale for Preprocessing Choices:
    ---------------------------------
    - Case Preservation: Maintains proper nouns and acronyms
    - Contraction Handling: Preserves French grammatical structure
    - Minimal Punctuation Removal: Keeps sentence boundaries and quotes
    - Lemmatization: Uses spaCy's context-aware lemmatization
    
    Args:
        text (str): Raw French text to process
        
    Returns:
        dict: Processed text at different levels:
            - article (str): Full processed text
            - paragraphs (list): List of processed paragraphs
            - sentences (list): List of processed sentences
    """
    if not text:
        return {
            "article": "",
            "paragraphs": [],
            "sentences": []
        }
    
    # Initial text cleaning with enhanced French support
    text = (text.strip()
        .replace('\xa0', ' ')        # Replace non-breaking spaces
        .replace('  ', ' ')          # Replace double spaces
        .replace('«', '"')           # Normalize French quotes
        .replace('»', '"')
        .replace('"', '"')           # Normalize other quotes
        .replace('"', '"')
        .replace('…', '...')         # Normalize ellipsis
        .replace('–', '-')           # Normalize dashes
        .replace('—', '-')
        .replace(''', "'")           # Normalize apostrophes
        .replace(''', "'"))

    # Enhanced French contraction handling
    # Preserve common French contractions while handling special cases
    contraction_patterns = [
        (r"([ldnmtsjc])'([a-zéèêëàâäôöûüç])", r"\1'\2"),  # l'exemple, d'accord, n'est
        (r"qu'([a-zéèêëàâäôöûüç])", r"qu'\1"),            # qu'il, qu'elle
        (r"jusqu'([aàeéèêëiîïoôuù])", r"jusqu'\1"),        # jusqu'à
        (r"presqu'([îi]le)", r"presqu'\1"),                # presqu'île
        (r"puisqu'([aàeéèêëiîïoôuù])", r"puisqu'\1"),     # puisqu'il
        (r"lorsqu'([aàeéèêëiîïoôuù])", r"lorsqu'\1"),     # lorsqu'on
    ]
    
    for pattern, replacement in contraction_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Process with spaCy
    doc = nlp(text)
    
    def process_tokens(tokens):
        """
        Process tokens with French-specific handling.
        
        - Preserves case for proper nouns and acronyms
        - Keeps important French particles
        - Removes numbers and symbols
        """
        processed_tokens = []
        for token in tokens:
            # Skip unwanted tokens
            if (token.is_stop or token.is_punct or token.is_space or
                token.like_num or token.pos_ in ['SPACE', 'SYM'] or
                len(token.text.strip()) <= 1):
                continue
                
            # Handle proper nouns and acronyms
            if token.pos_ == 'PROPN' or token.text.isupper():
                processed_tokens.append(token.text)  # Preserve original case
            else:
                processed_tokens.append(token.lemma_.lower())  # Lowercase lemma for regular words
                
        return processed_tokens
    
    # Process at article level
    article_tokens = process_tokens(doc)
    
    # Process at paragraph level with enhanced handling
    paragraphs = []
    for para in text.split('\n\n'):
        if para.strip():
            para_doc = nlp(para)
            para_tokens = process_tokens(para_doc)
            if para_tokens:  # Only add non-empty paragraphs
                paragraphs.append(" ".join(para_tokens))
    
    # Process at sentence level with better boundary detection
    sentences = []
    for sent in doc.sents:
        sent_tokens = process_tokens(sent)
        if sent_tokens:  # Only add non-empty sentences
            sentences.append(" ".join(sent_tokens))
    
    return {
        "article": " ".join(article_tokens),
        "paragraphs": paragraphs,
        "sentences": sentences
    }

semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

async def fetch_data_async(session, url, max_retries=3, timeout=30):
    """
    Fetch data from the Omeka API with authentication, retries, and timeout handling.
    
    This function handles authenticated requests to the Omeka API with:
    - Rate limiting through semaphore
    - Automatic retries for failed requests
    - Configurable timeout
    - Exponential backoff between retries
    
    Args:
        session (aiohttp.ClientSession): Active HTTP session
        url (str): URL to fetch data from
        max_retries (int): Maximum number of retry attempts
        timeout (int): Timeout in seconds for each request
        
    Returns:
        dict: JSON response data or None if request fails
        
    Notes:
        - Uses exponential backoff between retries
        - Logs detailed error information
        - Handles various types of connection errors
    """
    # Add key_identity and key_credential as URL parameters
    if '?' in url:
        auth_url = f"{url}&key_identity={OMEKA_KEY_IDENTITY}&key_credential={OMEKA_KEY_CREDENTIAL}"
    else:
        auth_url = f"{url}?key_identity={OMEKA_KEY_IDENTITY}&key_credential={OMEKA_KEY_CREDENTIAL}"
    
    headers = {
        'Authorization': f'ApiKey {OMEKA_KEY_IDENTITY}:{OMEKA_KEY_CREDENTIAL}'
    }
    
    for attempt in range(max_retries):
        try:
            async with semaphore:
                timeout_ctx = aiohttp.ClientTimeout(total=timeout)
                async with session.get(auth_url, headers=headers, timeout=timeout_ctx) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except asyncio.TimeoutError:
            wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
            logging.warning(f"Timeout on attempt {attempt + 1}/{max_retries} for {url}. Waiting {wait_time} seconds before retry...")
            await asyncio.sleep(wait_time)
            
        except aiohttp.ClientError as e:
            wait_time = (2 ** attempt) * 2
            logging.warning(f"Network error on attempt {attempt + 1}/{max_retries} for {url}: {str(e)}. Waiting {wait_time} seconds before retry...")
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            logging.error(f"Unexpected error fetching {url}: {str(e)}")
            return None
    
    logging.error(f"Failed to fetch data from {url} after {max_retries} attempts")
    return None

async def fetch_ids_from_item_async(session, url):
    """
    Fetch and validate article IDs from an Omeka item.
    
    This function:
    1. Fetches the main item data
    2. Extracts related article URLs
    3. Validates each URL in parallel using asyncio.gather()
    4. Tracks progress with tqdm
    
    Args:
        session (aiohttp.ClientSession): Active HTTP session
        url (str): URL of the main item to fetch related articles from
        
    Returns:
        list: Valid article URLs that were successfully validated
    """
    logging.info(f"Starting to fetch IDs from {url}")
    data = await fetch_data_async(session, url)
    
    if not data or '@reverse' not in data or 'dcterms:subject' not in data['@reverse']:
        logging.error(f"Failed to fetch or parse data from {url}")
        return []
    
    potential_urls = [item['@id'] for item in data['@reverse']['dcterms:subject']]
    logging.info(f"Found {len(potential_urls)} potential URLs to process")
    
    # Create progress bar for URL validation
    pbar = tqdm(total=len(potential_urls), desc="Validating articles", unit="item")
    
    async def validate_url(url):
        """Helper function to validate a single URL."""
        item_data = await fetch_data_async(session, url)
        pbar.update(1)
        if not item_data:
            logging.warning(f"Failed to fetch data for URL: {url}")
            return None
            
        if '@type' in item_data and 'bibo:Article' in item_data['@type']:
            title = item_data.get('o:title', 'Untitled')
            logging.info(f"Found article: '{title}' at {url}")
            return url
        return None
    
    # Create tasks for all URLs and gather them
    tasks = [validate_url(url) for url in potential_urls]
    results = await asyncio.gather(*tasks)
    
    # Filter out None values and close progress bar
    article_urls = [url for url in results if url is not None]
    pbar.close()
    
    logging.info(f"Processing complete: Found {len(article_urls)} articles out of {len(potential_urls)} total items")
    return article_urls

def process_article_content(article_data):
    """
    Process and enrich article content with NLP analysis.
    
    This function:
    1. Extracts content from article data
    2. Applies text preprocessing to the content
    3. Adds processed text back to the article data
    
    Args:
        article_data (dict): Raw article data from API
        
    Returns:
        dict: Enriched article data with processed text fields
        
    Notes:
        - Handles both list and dict content formats
        - Preserves original content while adding processed version
    """
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
    """
    Process multiple article URLs concurrently using asyncio.gather().
    
    This function:
    1. Creates a single session for all requests
    2. Processes URLs concurrently with rate limiting via semaphore
    3. Tracks progress and success/failure counts
    
    Args:
        urls (list): List of article URLs to process
        
    Returns:
        tuple: Contains:
            - list: Processed article data
            - int: Number of successful fetches
            - int: Number of failed fetches
    """
    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(urls), desc="Processing articles", unit="article")
        
        async def process_single_url(url):
            """Helper function to process a single URL."""
            data = await fetch_data_async(session, url)
            pbar.update(1)
            if data:
                return process_article_content(data)
            return None
        
        # Create tasks for all URLs and gather them
        tasks = [process_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        # Filter out None values and count successes/failures
        all_data = [result for result in results if result is not None]
        successful_fetches = len(all_data)
        failed_fetches = len(urls) - successful_fetches
        
        pbar.close()
        return all_data, successful_fetches, failed_fetches

async def get_item_title(session, item_url):
    """
    Fetch the title of an Omeka item.
    
    Args:
        session (aiohttp.ClientSession): Active HTTP session
        item_url (str): URL of the item
        
    Returns:
        str: The item's title or a default title if not found
    """
    try:
        data = await fetch_data_async(session, item_url)
        if data and 'o:title' in data:
            title = data['o:title']
            # Clean the title to make it suitable for a filename
            title = re.sub(r'[<>:"/\\|?*]', '_', title)  # Replace invalid filename characters
            title = re.sub(r'\s+', '_', title)  # Replace spaces with underscores
            return title
    except Exception as e:
        logging.error(f"Failed to fetch item title: {str(e)}")
    
    return f"omeka_item_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

async def main_async():
    """
    Main asynchronous execution function.
    
    This function orchestrates the entire data collection and processing pipeline:
    1. Gets item ID from user input
    2. Sets up logging with item ID
    3. Initializes API connection
    4. Fetches article URLs
    5. Processes articles concurrently
    6. Saves results to JSON in data directory
    7. Logs statistics and progress
    """
    # Get item ID from user
    item_id = input("Please enter the Omeka item ID: ").strip()
    
    # Set up logging with item ID
    log_path = setup_logging(item_id)
    logging.info(f"Starting data collection for Omeka item {item_id}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    base_item_url = f"{OMEKA_BASE_URL}/items/{item_id}"
    # Add authentication parameters to the initial URL
    item_url = f"{base_item_url}?key_identity={OMEKA_KEY_IDENTITY}&key_credential={OMEKA_KEY_CREDENTIAL}"
    
    async with aiohttp.ClientSession() as session:
        # Get the item title for the output filename
        item_title = await get_item_title(session, item_url)
        logging.info(f"Processing item: {item_title}")
        
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

        # Save all data to a JSON file in the data directory
        output_file = os.path.join(data_dir, f'{item_title}.json')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=4)
            logging.info(f"Data successfully saved to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save data to file: {str(e)}")

def main():
    """
    Entry point for the script.
    
    This function:
    1. Sets up the async event loop
    2. Runs the main async function
    3. Handles any top-level exceptions
    """
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 