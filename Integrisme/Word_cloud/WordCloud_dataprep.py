import pandas as pd
from collections import Counter
import json
from tqdm import tqdm
import os
from transformers import CamembertTokenizer, CamembertModel
import torch
import re

# Initialize CamemBERT
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertModel.from_pretrained('camembert-base')

# Expanded French stopwords
french_stopwords = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'à', 'au', 'aux',
    'et', 'ou', 'mais', 'donc', 'car', 'ni', 'que', 'qui', 'quoi', 'dont',
    'où', 'dans', 'sur', 'sous', 'avec', 'sans', 'pour', 'par', 'en', 'vers',
    'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'notre', 'votre', 'leur',
    'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se',
    'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'venir', 'falloir', 'pouvoir',
    'plus', 'moins', 'très', 'bien', 'mal', 'peu', 'trop', 'beaucoup', 'aussi', 'el',
    'fait', 'fois', 'cas', 'chose', 'rien', 'tout', 'tous', 'toute', 'toutes',
    'autre', 'autres', 'même', 'mêmes', 'après', 'avant', 'depuis', 'pendant',
    'selon', 'ainsi', 'alors', 'encore', 'toujours', 'jamais', 'lors', 'dont',
    'cela', 'ceci', 'celui', 'celle', 'ceux', 'celles'
}

def load_data(url):
    """Load data from URL."""
    return pd.read_csv(url)

def filter_integrisme_articles(df):
    """Filter articles mentioning 'intégrisme'."""
    return df[df['dcterms:subject'].fillna('').str.contains('Intégrisme', case=False)]

def clean_text(text):
    """Initial text cleaning before tokenization."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace various types of apostrophes and quotes
    text = text.replace(''', "'").replace(''', "'").replace('`', "'")
    text = text.replace('"', '"').replace('"', '"').replace('«', '"').replace('»', '"')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove digits and digit-word combinations
    text = re.sub(r'\w*\d\w*', '', text)
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep French accents
    text = re.sub(r'[^\w\s\'\-àáâãäçèéêëìíîïñòóôõöùúûüýÿ]', ' ', text)
    
    return text.strip()

def merge_split_words(tokens):
    """Merge commonly split words."""
    merged_tokens = []
    i = 0
    while i < len(tokens):
        current_token = tokens[i]
        
        # List of common split words to merge
        if i + 1 < len(tokens):
            next_token = tokens[i + 1]
            
            # Add more cases as needed
            if current_token == "aujourd" and next_token == "hui":
                merged_tokens.append("aujourd'hui")
                i += 2
            elif current_token == "c" and next_token == "est":
                merged_tokens.append("c'est")
                i += 2
            elif current_token == "d" and next_token == "abord":
                merged_tokens.append("d'abord")
                i += 2
            else:
                merged_tokens.append(current_token)
                i += 1
        else:
            merged_tokens.append(current_token)
            i += 1
    
    return merged_tokens

def preprocess_text(text):
    """Preprocess text using CamemBERT tokenizer."""
    # Initial cleaning
    text = clean_text(text)
    
    # Tokenize the text
    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    # Get token representations
    with torch.no_grad():
        outputs = model(**encoded)
    
    # Get the tokens
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    
    # Clean tokens
    cleaned_tokens = [
        token.replace('▁', '').lower()
        for token in tokens
        if (token.replace('▁', '').isalpha() 
            and not token.startswith('<')
            and not token.startswith('##')
            and len(token.replace('▁', '')) > 1)
    ]
    
    # Merge split words
    merged_tokens = merge_split_words(cleaned_tokens)
    
    # Remove stopwords
    final_tokens = [token for token in merged_tokens if token not in french_stopwords]
    
    return final_tokens

def generate_word_frequencies(tokens_list, max_words=200):
    """Generate word frequencies from list of tokens.
    
    Args:
        tokens_list: List of token lists from processed articles
        max_words: Maximum number of words to include in output (default: 100)
    """
    # Flatten the list of tokens and count frequencies
    all_tokens = [token for sublist in tokens_list for token in sublist]
    word_freq = Counter(all_tokens)
    
    # Convert to dictionary format, limiting to max_words
    return dict(word_freq.most_common(max_words))

def save_to_json(word_frequencies, output_path):
    """Save word frequencies to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(word_frequencies, f, ensure_ascii=False, indent=2)

def main():
    """Main function to generate word frequency JSON."""
    # URL for the data
    url = "https://raw.githubusercontent.com/fmadore/Islam-West-Africa-Collection/main/Metadata/CSV/newspaper_articles.csv"
    
    # Load data
    df = load_data(url)
    
    # Filter articles mentioning "intégrisme"
    integrisme_articles = filter_integrisme_articles(df)
    
    # Preprocess text in "bibo:content" with progress tracking
    tqdm.pandas(desc="Processing articles")
    integrisme_articles['processed_tokens'] = integrisme_articles['bibo:content'].fillna('').progress_apply(preprocess_text)
    
    # Generate word frequencies (top 100 words)
    word_frequencies = generate_word_frequencies(integrisme_articles['processed_tokens'], max_words=100)
    
    # Save to JSON file in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'word_frequencies.json')
    save_to_json(word_frequencies, output_path)
    print(f"Word frequencies saved to {output_path}")

if __name__ == "__main__":
    main() 