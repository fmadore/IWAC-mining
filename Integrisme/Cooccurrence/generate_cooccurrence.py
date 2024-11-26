import json
import pandas as pd
import numpy as np
from collections import defaultdict
import spacy
from pathlib import Path

def load_integrisme_data():
    """Load the integrisme data from JSON file"""
    # Get the current script's directory
    current_dir = Path(__file__).parent
    # Go up one level to the Integrisme directory and get the data file
    data_file = current_dir.parent / 'integrisme_data.json'
    
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_top_terms(word_frequencies_path, n_terms=50):
    """Load and get top N terms from word frequencies"""
    # Get the current script's directory
    current_dir = Path(__file__).parent
    # Construct path to word frequencies file
    word_frequencies_file = current_dir.parent / 'Word_cloud/data/word_frequencies.json'
    
    with open(word_frequencies_file, 'r', encoding='utf-8') as f:
        word_freq = json.load(f)
    return list(dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:n_terms]).keys())

def calculate_cooccurrence(articles, top_terms):
    """Calculate co-occurrence matrix for top terms"""
    n = len(top_terms)
    term_to_idx = {term: i for i, term in enumerate(top_terms)}
    matrix = np.zeros((n, n))
    
    for article in articles:
        # Extract content from the nested structure
        try:
            # Get the first item from bibo:content list and its @value
            content = article.get('bibo:content', [{}])[0].get('@value', '').lower()
            
            # Create a binary vector for terms present in this article
            present_terms = [term for term in top_terms if term in content]
            
            # Update co-occurrence matrix
            for i, term1 in enumerate(present_terms):
                for term2 in present_terms[i:]:
                    idx1, idx2 = term_to_idx[term1], term_to_idx[term2]
                    matrix[idx1][idx2] += 1
                    if idx1 != idx2:
                        matrix[idx2][idx1] += 1
                        
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Warning: Could not process article: {e}")
            continue
    
    return matrix.tolist()

def generate_matrix_data():
    """Generate the matrix data in the required format"""
    # Load data
    articles = load_integrisme_data()
    
    # Get top terms from word frequencies
    top_terms = get_top_terms(word_frequencies_path=None)  # path is handled inside the function
    
    # Calculate co-occurrence matrix
    matrix = calculate_cooccurrence(articles, top_terms)
    
    # Prepare the output data structure
    output_data = {
        "nodes": [{"id": i, "name": term} for i, term in enumerate(top_terms)],
        "links": []
    }
    
    # Convert matrix to links
    n = len(top_terms)
    for i in range(n):
        for j in range(n):  # Changed to include all combinations
            if matrix[i][j] > 0:
                output_data["links"].append({
                    "source": i,
                    "target": j,
                    "value": matrix[i][j]
                })
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    # Save the data
    output_file = output_dir / 'cooccurrence.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_matrix_data() 