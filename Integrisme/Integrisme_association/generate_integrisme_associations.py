import json
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

def load_integrisme_data():
    """Load the integrisme data from JSON file"""
    current_dir = Path(__file__).parent
    data_file = current_dir.parent / 'integrisme_data.json'
    
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_french_stopwords():
    """Get a set of French stopwords"""
    stopwords = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'ce', 'ces', 'cette',
        'il', 'elle', 'ils', 'elles', 'nous', 'vous', 'je', 'tu', 'on',
        'et', 'ou', 'où', 'mais', 'donc', 'car', 'ni', 'que', 'qui', 'quoi', 'dont',
        'dans', 'sur', 'sous', 'avec', 'sans', 'chez', 'pour', 'par', 'en',
        'au', 'aux', 'à', 'vers', 'depuis',
        'son', 'sa', 'ses', 'leur', 'leurs', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
        'ce', 'cet', 'cette', 'ces', 'être', 'avoir', 'faire', 'dire', 'aller',
        'tout', 'tous', 'toute', 'toutes', 'autre', 'autres', 'même', 'mêmes',
        'plus', 'moins', 'très', 'bien', 'mal', 'peu', 'trop',
        'ici', 'là', 'cela', 'ceci', 'celui', 'celle', 'ceux', 'celles',
        'alors', 'ainsi', 'car', 'donc', 'ensuite', 'puis',
        'comme', 'comment', 'pourquoi', 'quand', 'après'
    }
    return stopwords

def calculate_word_associations(articles, window_size=50):
    """Calculate associations with 'intégrisme' and 'intégriste'"""
    target_words = {'intégrisme', 'intégriste'}
    stopwords = get_french_stopwords()
    associations = defaultdict(int)
    context_counts = defaultdict(lambda: {'total': 0, 'by_target': defaultdict(int)})
    
    for article in articles:
        try:
            content = article.get('processed_text', {}).get('article', '').lower()
            if not content:
                content = article.get('bibo:content', [{}])[0].get('@value', '').lower()
        except:
            content = article.get('bibo:content', [{}])[0].get('@value', '').lower()
            
        words = content.split()
        
        # Find associations using sliding window
        for i, word in enumerate(words):
            if word in target_words:
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_word = words[j]
                        if (len(context_word) > 2 and 
                            context_word.isalpha() and 
                            context_word not in stopwords):
                            associations[context_word] += 1
                            context_counts[context_word]['total'] += 1
                            context_counts[context_word]['by_target'][word] += 1
    
    return associations, context_counts

def generate_association_data(min_occurrences=5, top_n=50):
    """Generate word association data"""
    articles = load_integrisme_data()
    associations, context_counts = calculate_word_associations(articles)
    
    # Convert to list and sort by frequency
    word_data = []
    for word, count in associations.items():
        if count >= min_occurrences:
            word_data.append({
                'word': word,
                'total_associations': count,
                'with_integrisme': context_counts[word]['by_target']['intégrisme'],
                'with_integriste': context_counts[word]['by_target']['intégriste']
            })
    
    # Sort by total associations
    word_data.sort(key=lambda x: x['total_associations'], reverse=True)
    word_data = word_data[:top_n]
    
    # Save the data
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'integrisme_associations.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(word_data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print(f"Top {min(top_n, len(word_data))} associated words:")
    for item in word_data[:20]:
        print(f"{item['word']}: {item['total_associations']} total "
              f"(intégrisme: {item['with_integrisme']}, "
              f"intégriste: {item['with_integriste']})")

if __name__ == "__main__":
    generate_association_data(min_occurrences=5, top_n=50) 