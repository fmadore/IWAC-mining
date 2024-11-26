import json
import requests
from tqdm import tqdm
import logging
from datetime import datetime
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set up logging
log_filename = f'spatial_mapping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_path = os.path.join(script_dir, log_filename)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

def load_integrisme_data():
    """Load the integrisme data from JSON file."""
    try:
        # Go up one directory level to access integrisme_data.json
        json_path = os.path.join(os.path.dirname(script_dir), 'integrisme_data.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading integrisme data: {str(e)}")
        return []

def extract_spatial_ids(articles):
    """Extract all spatial IDs from articles with their frequency."""
    spatial_counts = {}
    for article in articles:
        if 'dcterms:subject' in article:
            # Check if article mentions intégrisme
            has_integrisme = any(
                subject.get('@id', '').endswith('/59') 
                for subject in article['dcterms:subject']
            )
            if has_integrisme and 'dcterms:spatial' in article:
                for location in article['dcterms:spatial']:
                    if '@id' in location:
                        spatial_counts[location['@id']] = spatial_counts.get(location['@id'], 0) + 1
    
    logging.info(f"Found {len(spatial_counts)} locations with mentions")
    return spatial_counts

def get_coordinates(location_url):
    """Fetch coordinates for a location from its API URL."""
    try:
        response = requests.get(location_url)
        data = response.json()
        
        if 'curation:coordinates' in data:
            coords_str = data['curation:coordinates'][0]['@value']
            lat, lon = map(float, coords_str.split(','))
            return {
                'name': data.get('o:title', 'Unknown'),
                'coordinates': [lon, lat]  # GeoJSON uses [longitude, latitude]
            }
        else:
            logging.warning(f"No coordinates found for location: {data.get('o:title', 'Unknown')} ({location_url})")
            return None
            
    except Exception as e:
        logging.error(f"Error fetching coordinates for {location_url}: {str(e)}")
        return None

def create_geojson(locations_data):
    """Create GeoJSON from locations data."""
    features = []
    
    for location in locations_data:
        if location is not None:
            feature = {
                "type": "Feature",
                "properties": {
                    "name": location['name'],
                    "mentions": location['mentions']
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": location['coordinates']
                }
            }
            features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson

def save_geojson(geojson_data, filename='integrisme_locations.geojson'):
    """Save GeoJSON data to file."""
    try:
        output_path = os.path.join(script_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, ensure_ascii=False, indent=2)
        logging.info(f"GeoJSON data saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving GeoJSON data: {str(e)}")
        return False

def main():
    logging.info("Starting spatial mapping process")
    
    # Load articles data
    articles = load_integrisme_data()
    if not articles:
        logging.error("No articles data found. Exiting.")
        return
    
    logging.info(f"Loaded {len(articles)} articles")
    
    # Extract spatial IDs with counts
    spatial_counts = extract_spatial_ids(articles)
    if not spatial_counts:
        logging.error("No spatial data found. Exiting.")
        return
    
    # Get coordinates for each location
    locations = []
    for url, count in tqdm(spatial_counts.items(), desc="Fetching coordinates"):
        location_data = get_coordinates(url)
        if location_data:
            location_data['mentions'] = count
            locations.append(location_data)
    
    logging.info(f"Successfully retrieved coordinates for {len(locations)} locations")
    
    # Create GeoJSON
    geojson_data = create_geojson(locations)
    
    # Save GeoJSON file
    if save_geojson(geojson_data):
        logging.info("Process completed successfully")
    else:
        logging.error("Failed to save GeoJSON data")

if __name__ == "__main__":
    main()