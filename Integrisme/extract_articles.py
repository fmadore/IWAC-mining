"""
This module extracts articles from a JSON file and organizes them by publisher.
It processes articles from various Ivorian newspapers and saves them into separate text files,
one for each publisher. The module handles data extraction, cleaning, and file organization
while maintaining a consistent output format.
"""

import json
import os

def extract_articles_by_publisher(json_file, output_dir="extracted_articles"):
    """
    Extract and organize articles from a JSON file by their publisher.
    
    Args:
        json_file (str): Path to the input JSON file containing article data
        output_dir (str): Directory where the extracted articles will be saved (default: "extracted_articles")
    
    The function performs the following steps:
    1. Creates separate text files for each publisher
    2. Processes the JSON data to extract article information
    3. Writes articles to their respective publisher files with consistent formatting
    4. Handles file operations safely with proper encoding and error handling
    """
    # Define list of publishers to extract articles for
    publishers = [
        "Fraternité Matin",
        "La Voie",
        "Le Patriote", 
        "Le Jour",
        "Le Nouvel Horizon",
        "Notre Voie",
        "Notre Temps",
        "Fraternité Hebdo",
        "Alif",
        "Plume Libre"
    ]
    
    # Set up file paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, json_file)
    output_dir_path = os.path.join(script_dir, output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Create file handles for each publisher with consistent naming convention
    file_handles = {}
    for publisher in publishers:
        filename = f"{publisher.lower().replace(' ', '_')}_articles.txt"
        filepath = os.path.join(output_dir_path, filename)
        file_handles[publisher] = open(filepath, 'w', encoding='utf-8')
    
    try:
        # Load and parse the JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each article in the dataset
        for article in data:
            # Check if article has publisher information
            if 'dcterms:publisher' in article:
                for pub in article['dcterms:publisher']:
                    publisher = pub.get('display_title')
                    if publisher in publishers:
                        # Extract article metadata
                        title = article.get('o:title', 'No title')
                        
                        # Extract and format publication date
                        date = None
                        if 'dcterms:date' in article:
                            for d in article['dcterms:date']:
                                if '@value' in d:
                                    date = d['@value']
                                    break
                        
                        # Extract article content
                        content = None
                        if 'bibo:content' in article:
                            for c in article['bibo:content']:
                                if '@value' in c:
                                    content = c['@value']
                                    break
                        
                        # Write article to file with consistent formatting
                        f = file_handles[publisher]
                        f.write(f"TITLE: {title}\n")
                        f.write(f"DATE: {date}\n")
                        f.write("CONTENT:\n")
                        if content:
                            f.write(f"{content}\n")
                        # Add clear separator between articles
                        f.write("\n" + "="*80 + "\n\n")
                        break  # Stop after finding first matching publisher
                    
    finally:
        # Ensure all files are properly closed
        for f in file_handles.values():
            f.close()

if __name__ == "__main__":
    # Execute the extraction when run as a script
    input_file = "integrisme_data.json"
    extract_articles_by_publisher(input_file) 