import json
import os
from collections import Counter

def load_data(json_path):
    """Load data from JSON file."""
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
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to input JSON file (going up one directory)
    input_path = os.path.join(os.path.dirname(script_dir), 'integrisme_data.json')
    
    # Load data
    articles = load_data(input_path)
    
    # Extract preprocessed texts
    processed_texts = extract_processed_text(articles)
    
    # Generate word frequencies
    word_frequencies = generate_word_frequencies(processed_texts, max_words=200)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to JSON file
    output_path = os.path.join(data_dir, 'word_frequencies.json')
    save_to_json(word_frequencies, output_path)
    print(f"Word frequencies saved to {output_path}")

if __name__ == "__main__":
    main() 