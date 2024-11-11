import json
import requests
import folium
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
    json_path = os.path.join(script_dir, 'integrisme_data.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_spatial_ids(articles):
    """Extract all spatial IDs from articles."""
    spatial_ids = set()
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
                        spatial_ids.add(location['@id'])
    return spatial_ids

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
                'coordinates': (lat, lon)
            }
        else:
            logging.warning(f"No coordinates found for location: {data.get('o:title', 'Unknown')} ({location_url})")
            return None
            
    except Exception as e:
        logging.error(f"Error fetching coordinates for {location_url}: {str(e)}")
        return None

def create_map(locations):
    """Create a folium map with the locations."""
    # Create map centered on average coordinates
    valid_coords = [(loc['coordinates'][0], loc['coordinates'][1]) 
                   for loc in locations if loc is not None]
    
    if not valid_coords:
        logging.error("No valid coordinates found to create map")
        return None
        
    center_lat = sum(lat for lat, _ in valid_coords) / len(valid_coords)
    center_lon = sum(lon for _, lon in valid_coords) / len(valid_coords)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    
    # Add markers for each location
    for loc in locations:
        if loc is not None:
            folium.Marker(
                location=loc['coordinates'],
                popup=loc['name'],
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
    
    return m

def main():
    logging.info("Starting spatial mapping process")
    
    # Load data
    articles = load_integrisme_data()
    logging.info(f"Loaded {len(articles)} articles")
    
    # Extract spatial IDs
    spatial_ids = extract_spatial_ids(articles)
    logging.info(f"Found {len(spatial_ids)} unique locations")
    
    # Get coordinates for each location
    locations = []
    for url in tqdm(spatial_ids, desc="Fetching coordinates"):
        location_data = get_coordinates(url)
        if location_data:
            locations.append(location_data)
    
    logging.info(f"Successfully retrieved coordinates for {len(locations)} locations")
    
    # Create map
    m = create_map(locations)
    if m:
        # Save map
        output_path = os.path.join(script_dir, 'integrisme_locations.html')
        m.save(output_path)
        logging.info(f"Map saved to {output_path}")
    else:
        logging.error("Failed to create map")

if __name__ == "__main__":
    main()
