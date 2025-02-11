"""
This module extracts articles from a JSON file and organizes them by publisher.
It processes articles from various Ivorian newspapers and saves them into separate text files,
one for each publisher. The module handles data extraction, cleaning, and file organization
while maintaining a consistent output format.
"""

import json
import os

def get_json_file_choice(data_dir):
    """
    Lists all JSON files in the specified directory and lets the user choose one.
    
    Args:
        data_dir (str): Path to the directory containing JSON files
        
    Returns:
        str: Name of the selected JSON file, or None if no files found or invalid selection
    """
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found in the data directory.")
        return None
    
    # Print available files with numbers
    print("\nAvailable JSON files:")
    for idx, file in enumerate(json_files, 1):
        print(f"{idx}. {file}")
    
    # Get user choice
    while True:
        try:
            choice = input("\nEnter the number of the file to process (or 'q' to quit): ")
            if choice.lower() == 'q':
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(json_files):
                return json_files[choice_idx]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")

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
    
    # Set up file paths relative to the workspace root
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(workspace_root, "data", json_file)
    output_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    
    print(f"Looking for JSON file at: {json_path}")  # Debug print
    
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
    # Set up the data directory path
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(workspace_root, "data")
    
    # Get user's choice of JSON file
    selected_file = get_json_file_choice(data_dir)
    
    if selected_file:
        print(f"\nProcessing file: {selected_file}")
        extract_articles_by_publisher(selected_file)
    else:
        print("No file selected. Exiting.") 