import pandas as pd
from collections import Counter
import json
from tqdm import tqdm
import os
import spacy
import re
from spacy.lang.fr.stop_words import STOP_WORDS

# Load French transformer model
nlp = spacy.load('fr_dep_news_trf')

# Combine spaCy's stopwords with custom ones
custom_stopwords = {
    # Words specific to newspaper articles and reporting
    'afp', 'reuters', 'ap', 'photo', 'photographe', 'journal', 'article',
    'lire', 'voir', 'dit', 'dire', 'faire', 'être', 'avoir',
    # Common verbs in news reporting
    'déclarer', 'affirmer', 'expliquer', 'indiquer', 'préciser', 'ajouter',
    'poursuivre', 'conclure', 'annoncer', 'rapporter', 'souligner',
    # Time-related words
    'année', 'mois', 'semaine', 'jour', 'hier', 'aujourd', 'hui', 'demain',
    'lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche',
    'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet',
    'août', 'septembre', 'octobre', 'novembre', 'décembre',
    # Numbers and quantities
    'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf', 'dix',
    'premier', 'première', 'second', 'seconde', 'dernier', 'dernière',
    # Common but uninformative words in news context
    'façon', 'manière', 'chose', 'fois', 'cas', 'exemple', 'partie',
    'moment', 'temps', 'heure', 'période',
}

# Combine all stopwords
french_stopwords = set(STOP_WORDS) | custom_stopwords

def load_data(url):
    """Load data from URL."""
    return pd.read_csv(url)

def filter_integrisme_articles(df):
    """Filter articles mentioning 'intégrisme'."""
    # Create an explicit copy of the filtered DataFrame
    return df[df['dcterms:subject'].fillna('').str.contains('Intégrisme', case=False)].copy()

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

def preprocess_text(text):
    """Preprocess text using spaCy for lemmatization."""
    # Initial cleaning
    text = clean_text(text)
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Get lemmatized tokens, filtering out stopwords and short words
    cleaned_words = [
        token.lemma_.lower()
        for token in doc
        if (
            token.lemma_.lower() not in french_stopwords 
            and len(token.lemma_) > 1
            and token.lemma_.isalpha()
            and not token.is_punct
            and not token.is_space
            and not token.is_digit
            and not token.like_num  # Catches written numbers like 'trois'
            and not token.is_currency
            and token.pos_ not in ['AUX', 'DET', 'PRON', 'ADP', 'SCONJ', 'CCONJ']
        )
    ]
    
    return cleaned_words

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
    
    # Filter articles mentioning "intégrisme" and create a copy
    integrisme_articles = filter_integrisme_articles(df)
    
    # Preprocess text in "bibo:content" with progress tracking
    tqdm.pandas(desc="Processing articles")
    integrisme_articles['processed_tokens'] = integrisme_articles['bibo:content'].fillna('').progress_apply(preprocess_text)
    
    # Generate word frequencies (top 200 words instead of 100)
    word_frequencies = generate_word_frequencies(integrisme_articles['processed_tokens'], max_words=200)
    
    # Save to JSON file in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'word_frequencies.json')
    save_to_json(word_frequencies, output_path)
    print(f"Word frequencies saved to {output_path}")

if __name__ == "__main__":
    main() 