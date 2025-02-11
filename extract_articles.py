"""
This module extracts articles from a JSON file and organizes them by publisher.
It processes articles from various newspapers and saves them into separate text files,
one for each publisher, plus a comprehensive file with all articles.
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
    1. Creates separate text files for each unique publisher found in the data
    2. Creates an all_articles.txt file containing all articles
    3. Processes the JSON data to extract article information
    4. Writes articles to their respective files with consistent formatting
    5. Handles file operations safely with proper encoding and error handling
    """
    
    # Set up file paths relative to the script location (now at root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "data", json_file)
    output_dir_path = os.path.join(script_dir, output_dir)
    
    print(f"Looking for JSON file at: {json_path}")
    print(f"Output directory will be: {output_dir_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir_path, exist_ok=True)
    
    # First pass: collect all unique publishers
    publishers = set()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for article in data:
            if 'dcterms:publisher' in article:
                for pub in article['dcterms:publisher']:
                    if 'display_title' in pub:
                        publishers.add(pub['display_title'])
    
        print(f"\nFound {len(publishers)} unique publishers:")
        for pub in sorted(publishers):
            print(f"- {pub}")
            
        # Create file handles for each publisher and the all_articles file
        file_handles = {}
        for publisher in publishers:
            filename = f"{publisher.lower().replace(' ', '_')}_articles.txt"
            filepath = os.path.join(output_dir_path, filename)
            file_handles[publisher] = open(filepath, 'w', encoding='utf-8')
        
        # Create the all_articles file
        all_articles_path = os.path.join(output_dir_path, "all_articles.txt")
        all_articles_file = open(all_articles_path, 'w', encoding='utf-8')
        
        # Second pass: process and write articles
        for article in data:
            if 'dcterms:publisher' in article:
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
                
                # Format article text
                article_text = f"TITLE: {title}\n"
                article_text += f"DATE: {date}\n"
                if content:
                    article_text += "CONTENT:\n"
                    article_text += f"{content}\n"
                article_text += "\n" + "="*80 + "\n\n"
                
                # Write to all_articles file
                all_articles_file.write(article_text)
                
                # Write to publisher-specific files
                for pub in article['dcterms:publisher']:
                    publisher = pub.get('display_title')
                    if publisher in publishers:
                        file_handles[publisher].write(article_text)
                        break
                        
    finally:
        # Ensure all files are properly closed
        for f in file_handles.values():
            f.close()
        if 'all_articles_file' in locals():
            all_articles_file.close()
            
        print(f"\nArticles have been extracted to: {output_dir_path}")
        print("Files created:")
        print(f"- all_articles.txt (contains all articles)")
        for publisher in sorted(publishers):
            filename = f"{publisher.lower().replace(' ', '_')}_articles.txt"
            print(f"- {filename}")

if __name__ == "__main__":
    # Set up the data directory path (now relative to script at root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    # Get user's choice of JSON file
    selected_file = get_json_file_choice(data_dir)
    
    if selected_file:
        print(f"\nProcessing file: {selected_file}")
        extract_articles_by_publisher(selected_file)
    else:
        print("No file selected. Exiting.") 