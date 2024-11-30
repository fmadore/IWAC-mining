import json
import os

def extract_articles_by_publisher(json_file, output_dir="extracted_articles"):
    # List of publishers to extract
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
    
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create full paths
    json_path = os.path.join(script_dir, json_file)
    output_dir_path = os.path.join(script_dir, output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Initialize file handles for each publisher
    file_handles = {}
    for publisher in publishers:
        filename = f"{publisher.lower().replace(' ', '_')}_articles.txt"
        filepath = os.path.join(output_dir_path, filename)
        file_handles[publisher] = open(filepath, 'w', encoding='utf-8')
    
    try:
        # Read the JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each article
        for article in data:
            # Check publisher
            if 'dcterms:publisher' in article:
                for pub in article['dcterms:publisher']:
                    publisher = pub.get('display_title')
                    if publisher in publishers:
                        # Extract title
                        title = article.get('o:title', 'No title')
                        
                        # Extract date
                        date = None
                        if 'dcterms:date' in article:
                            for d in article['dcterms:date']:
                                if '@value' in d:
                                    date = d['@value']
                                    break
                        
                        # Extract content
                        content = None
                        if 'bibo:content' in article:
                            for c in article['bibo:content']:
                                if '@value' in c:
                                    content = c['@value']
                                    break
                        
                        # Write to appropriate file
                        f = file_handles[publisher]
                        f.write(f"TITLE: {title}\n")
                        f.write(f"DATE: {date}\n")
                        f.write("CONTENT:\n")
                        if content:
                            f.write(f"{content}\n")
                        f.write("\n" + "="*80 + "\n\n")  # Separator between articles
                        break  # Break after finding the first matching publisher
                    
    finally:
        # Close all file handles
        for f in file_handles.values():
            f.close()

if __name__ == "__main__":
    input_file = "integrisme_data.json"
    extract_articles_by_publisher(input_file) 