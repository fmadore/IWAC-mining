import pandas as pd
from datetime import datetime
import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from asyncio import gather

# Load environment variables
load_dotenv()

OMEKA_BASE_URL = os.getenv('OMEKA_BASE_URL')
OMEKA_KEY_IDENTITY = os.getenv('OMEKA_KEY_IDENTITY')
OMEKA_KEY_CREDENTIAL = os.getenv('OMEKA_KEY_CREDENTIAL')

# Enhanced logging configuration
def setup_logging():
    """Configure logging with both file and console output."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, f'article_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

def get_keyword_mapping():
    """Return dictionary mapping concept IDs to keywords."""
    return {
        '59': 'Intégrisme',
        '21': 'Fondamentalisme islamique',
        '24': 'Islamisme',
        '63530': 'Radicalisation',
        '63372': 'Extrémisme',
        '63445': 'Obscurantisme',
        '33': 'Terrorisme',
        '63531': 'Djihadisme',
        '43': 'Salafisme'
    }

def get_newspaper_country_mapping():
    """Return dictionary mapping newspapers to their countries."""
    return {
        # Benin newspapers
        '24h au Bénin': 'Benin',
        'Agence Bénin Presse': 'Benin',
        'Banouto': 'Benin',
        'Bénin Intelligent': 'Benin',
        'Boulevard des Infos': 'Benin',
        'Daho-Express': 'Benin',
        'Ehuzu': 'Benin',
        'Fraternité': 'Benin',
        "L'Evénement Précis": 'Benin',
        'La Nation': 'Benin',
        'La Nouvelle Tribune': 'Benin',
        'Le Matinal': 'Benin',
        'Les Pharaons': 'Benin',
        'Matin Libre': 'Benin',
        
        # Burkina Faso newspapers
        'Burkina 24': 'Burkina Faso',
        'Carrefour africain': 'Burkina Faso',
        'FasoZine': 'Burkina Faso',
        "L'Evénement": 'Burkina Faso',
        "L'Observateur": 'Burkina Faso',
        "L'Observateur Paalga": 'Burkina Faso',
        'La Preuve': 'Burkina Faso',
        'Le Pays': 'Burkina Faso',
        'LeFaso.net': 'Burkina Faso',
        'Mutations': 'Burkina Faso',
        'San Finna': 'Burkina Faso',
        'Sidwaya': 'Burkina Faso',
        
        # Côte d'Ivoire newspapers
        'Agence Ivoirienne de Presse': 'Côte d\'Ivoire',
        'Alif': 'Côte d\'Ivoire',
        'Fraternité Hebdo': 'Côte d\'Ivoire',
        'Fraternité Matin': 'Côte d\'Ivoire',
        'Ivoire Dimanche': 'Côte d\'Ivoire',
        "L'Intelligent d'Abidjan": 'Côte d\'Ivoire',
        'La Voie': 'Côte d\'Ivoire',
        'Le Jour': 'Côte d\'Ivoire',
        'Le Jour Plus': 'Côte d\'Ivoire',
        'Le Nouvel Horizon': 'Côte d\'Ivoire',
        'Le Patriote': 'Côte d\'Ivoire',
        'Notre Temps': 'Côte d\'Ivoire',
        'Notre Voie': 'Côte d\'Ivoire',
        'Plume Libre': 'Côte d\'Ivoire',
        
        # Togo newspapers
        'Agence Togolaise de Presse': 'Togo',
        'Courrier du Golfe': 'Togo',
        'La Nouvelle Marche': 'Togo',
        'Togo-Presse': 'Togo'
    }

async def fetch_data_async(session, url):
    """Fetch data from Omeka S API with enhanced logging."""
    if '?' in url:
        auth_url = f"{url}&key_identity={OMEKA_KEY_IDENTITY}&key_credential={OMEKA_KEY_CREDENTIAL}"
    else:
        auth_url = f"{url}?key_identity={OMEKA_KEY_IDENTITY}&key_credential={OMEKA_KEY_CREDENTIAL}"
    
    try:
        async with session.get(auth_url) as response:
            response.raise_for_status()
            data = await response.json()
            logging.debug(f"Successfully fetched data from {url}")
            return data
    except aiohttp.ClientError as e:
        logging.error(f"Network error fetching {url}: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error for {url}: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching {url}: {str(e)}")
        return None

async def fetch_articles_for_concept(session, concept_id, keyword):
    """Fetch all articles related to a concept ID with progress tracking."""
    url = f"{OMEKA_BASE_URL}/items/{concept_id}"
    logging.info(f"Fetching articles for concept: {keyword} (ID: {concept_id})")
    
    data = await fetch_data_async(session, url)
    if not data:
        logging.error(f"Failed to fetch concept data for {keyword}")
        return []
    
    if '@reverse' not in data or 'dcterms:subject' not in data['@reverse']:
        logging.warning(f"No articles found for concept: {keyword}")
        return []
    
    related_items = data['@reverse']['dcterms:subject']
    logging.info(f"Found {len(related_items)} potential items for {keyword}")
    
    # Create tasks for all article fetches
    tasks = [fetch_data_async(session, item['@id']) for item in related_items]
    
    # Fetch all articles concurrently
    articles_data = await gather(*tasks)
    
    # Filter and process articles
    articles = []
    for article_data in articles_data:
        if article_data and 'bibo:Article' in article_data.get('@type', []):
            articles.append(article_data)
    
    logging.info(f"Successfully processed {len(articles)} articles for {keyword}")
    return articles

async def load_and_prepare_data():
    """Load data from Omeka S API and prepare it for analysis with enhanced logging."""
    keyword_mapping = get_keyword_mapping()
    newspaper_mapping = get_newspaper_country_mapping()
    
    stats = {
        'total_fetched': 0,
        'successful': 0,
        'failed': 0,
        'by_keyword': {},
        'unmapped_publishers': set()
    }
    
    all_articles = []
    async with aiohttp.ClientSession() as session:
        # Create tasks for all concepts
        tasks = [
            fetch_articles_for_concept(session, concept_id, keyword) 
            for concept_id, keyword in keyword_mapping.items()
        ]
        
        # Fetch all concepts' articles concurrently
        articles_by_keyword = await gather(*tasks)
        
        # Process results
        for (concept_id, keyword), articles in zip(keyword_mapping.items(), articles_by_keyword):
            stats['by_keyword'][keyword] = {'found': 0, 'processed': 0}
            stats['by_keyword'][keyword]['found'] = len(articles)
            
            for article in articles:
                try:
                    publisher = next((p.get('display_title') for p in article.get('dcterms:publisher', [])), None)
                    date_value = next((d.get('@value') for d in article.get('dcterms:date', [])), None)
                    
                    if publisher and date_value:
                        # Handle different date formats
                        if '/' in date_value:
                            # For ranges like "1995-01/1995-02", take the first date
                            date_value = date_value.split('/')[0]
                        
                        # For partial dates like "1995-01", append "-01" for the day
                        if len(date_value.split('-')) == 2:
                            date_value = f"{date_value}-01"
                        
                        if publisher not in newspaper_mapping:
                            stats['unmapped_publishers'].add(publisher)
                        
                        all_articles.append({
                            'publisher': publisher,
                            'date': date_value,
                            'keyword': keyword,
                            'country': newspaper_mapping.get(publisher)
                        })
                        stats['by_keyword'][keyword]['processed'] += 1
                        stats['successful'] += 1
                    else:
                        stats['failed'] += 1
                        logging.warning(f"Missing required data for article: publisher={publisher}, date={date_value}")
                except Exception as e:
                    stats['failed'] += 1
                    logging.error(f"Error processing article for {keyword}: {str(e)}")
    
    stats['total_fetched'] = stats['successful'] + stats['failed']
    
    # Log comprehensive statistics
    logging.info("\n=== Data Collection Statistics ===")
    logging.info(f"Total articles fetched: {stats['total_fetched']}")
    logging.info(f"Successfully processed: {stats['successful']}")
    logging.info(f"Failed to process: {stats['failed']}")
    logging.info("\nBy keyword:")
    for keyword, counts in stats['by_keyword'].items():
        logging.info(f"  {keyword}:")
        logging.info(f"    Found: {counts['found']}")
        logging.info(f"    Processed: {counts['processed']}")
    
    if stats['unmapped_publishers']:
        logging.warning("\nUnmapped publishers found:")
        for publisher in sorted(stats['unmapped_publishers']):
            logging.warning(f"  - {publisher}")
    
    # Create and process DataFrame
    df = pd.DataFrame(all_articles)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    
    # Log date range information
    logging.info(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    logging.info(f"Total unique publishers: {df['publisher'].nunique()}")
    logging.info(f"Total unique countries: {df['country'].nunique()}")
    
    return df

def prepare_combined_yearly_counts(df):
    """Prepare combined yearly counts data with enhanced validation."""
    keywords = list(get_keyword_mapping().values())
    
    # Add detailed logging for Intégrisme
    integrisme_total = len(df[df['keyword'] == 'Intégrisme'])
    logging.info(f"\nValidating Intégrisme data:")
    logging.info(f"Total Intégrisme articles in DataFrame: {integrisme_total}")
    
    # Log distribution by year
    yearly_dist = df[df['keyword'] == 'Intégrisme'].groupby('year').size()
    logging.info("\nIntégrisme articles by year:")
    for year, count in yearly_dist.items():
        logging.info(f"  {year}: {count} articles")
    
    # Log distribution by country
    country_dist = df[df['keyword'] == 'Intégrisme'].groupby('country').size()
    logging.info("\nIntégrisme articles by country:")
    for country, count in country_dist.items():
        logging.info(f"  {country}: {count} articles")
    
    # Check for any null values
    null_dates = df[df['keyword'] == 'Intégrisme']['date'].isnull().sum()
    null_countries = df[df['keyword'] == 'Intégrisme']['country'].isnull().sum()
    logging.info(f"\nNull values check:")
    logging.info(f"  Articles with null dates: {null_dates}")
    logging.info(f"  Articles with null countries: {null_countries}")
    
    # Get all unique years from the data
    all_years = sorted(df['year'].unique())
    
    # Initialize list to store results
    results = []
    total_processed = 0
    
    # For each year and keyword combination
    for year in all_years:
        if pd.isna(year):
            logging.warning(f"Skipping records with null year value")
            continue
            
        year_data = {}
        year_data['year'] = int(year)  # Convert to int for JSON serialization
        year_data['keywords'] = []
        
        for keyword in keywords:
            # Filter data for this year and keyword
            mask = (df['year'] == year) & (df['keyword'] == keyword)
            articles = df[mask]
            
            if keyword == 'Intégrisme':
                total_processed += len(articles)
                if len(articles) > 0:
                    logging.debug(f"Processing {len(articles)} Intégrisme articles for year {year}")
            
            if len(articles) > 0:
                # Group by country and count
                country_counts = articles.groupby('country').size().to_dict()
                
                keyword_data = {
                    'keyword': keyword,
                    'total_count': len(articles),
                    'countries': [
                        {'name': country, 'count': count}
                        for country, count in country_counts.items()
                        if pd.notna(country)  # Filter out NaN countries
                    ]
                }
                year_data['keywords'].append(keyword_data)
            else:
                # Include zero counts
                year_data['keywords'].append({
                    'keyword': keyword,
                    'total_count': 0,
                    'countries': []
                })
        
        results.append(year_data)
    
    logging.info(f"\nValidation summary:")
    logging.info(f"Total Intégrisme articles in raw data: {integrisme_total}")
    logging.info(f"Total Intégrisme articles processed in yearly counts: {total_processed}")
    
    # Validate the final JSON structure
    integrisme_in_json = sum(
        keyword_data['total_count']
        for year_data in results
        for keyword_data in year_data['keywords']
        if keyword_data['keyword'] == 'Intégrisme'
    )
    logging.info(f"Total Intégrisme articles in final JSON: {integrisme_in_json}")
    
    return results

async def main():
    """Main function to run the analysis."""
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting article comparison analysis")
    
    try:
        # Load and prepare data
        df = await load_and_prepare_data()
        
        # Prepare combined yearly counts
        yearly_counts = prepare_combined_yearly_counts(df)
        
        # Save the data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_output_path = os.path.join(script_dir, 'yearly_counts.json')
        
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(yearly_counts, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Analysis complete. Results saved to '{json_output_path}'")
        logging.info(f"Log file available at: {log_file}")
        
    except Exception as e:
        logging.error(f"Fatal error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 