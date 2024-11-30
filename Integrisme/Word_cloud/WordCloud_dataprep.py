import json
import os
import requests
from collections import Counter

def load_data(json_path):
    """Load data from JSON file or URL."""
    if json_path.startswith('http'):
        # Load from URL
        response = requests.get(json_path)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    else:
        # Load from local file
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def extract_processed_text(articles):
    """Extract preprocessed text from articles."""
    processed_texts = []
    
    for article in articles:
        # Get the content if it exists
        content = article.get('bibo:content', [{}])[0]
        
        # Get the processed text if it exists
        processed_text = content.get('processed_text', {})
        
        # Get the article text if it exists
        article_text = processed_text.get('article', '')
        
        if article_text:
            processed_texts.append(article_text)
            
    return processed_texts

def generate_word_frequencies(processed_texts, max_words=200):
    """Generate word frequencies from processed texts.
    
    Args:
        processed_texts: List of preprocessed article texts
        max_words: Maximum number of words to include in output (default: 200)
    """
    # Split each text into words and create a flat list
    all_words = []
    for text in processed_texts:
        words = text.split()
        all_words.extend(words)
    
    # Count frequencies
    word_freq = Counter(all_words)
    
    # Convert to dictionary format, limiting to max_words
    return dict(word_freq.most_common(max_words))

def save_to_json(word_frequencies, output_path):
    """Save word frequencies to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(word_frequencies, f, ensure_ascii=False, indent=2)

def main():
    """Main function to generate word frequency JSON."""
    # GitHub raw URL for the data
    github_url = "https://github.com/fmadore/Mining_IWAC/raw/refs/heads/main/Integrisme/Word_cloud/data/word_frequencies.json"
    
    try:
        # Load data from GitHub
        word_frequencies = load_data(github_url)
        
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(script_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to local JSON file (optional - if you want to keep a local copy)
        output_path = os.path.join(data_dir, 'word_frequencies.json')
        save_to_json(word_frequencies, output_path)
        print(f"Word frequencies saved to {output_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from GitHub: {e}")
        # Optionally fall back to local file if GitHub fetch fails
        print("Falling back to local file...")
        # Your existing local file loading code here

if __name__ == "__main__":
    main() 