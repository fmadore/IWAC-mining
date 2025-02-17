"""
Script to generate co-occurrence matrices and network data for term relationships in articles.

This script analyzes the relationships between terms in a corpus of articles by:
1. Loading article data from JSON
2. Identifying important terms using TF-IDF scoring
3. Calculating co-occurrence matrices at different text window levels (article, paragraph, sentence)
4. Generating network data for visualization

The output is a JSON file containing network data for D3.js visualization.
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

def load_integrisme_data():
    """
    Load the integrisme article data from JSON file.
    
    Returns:
        list: List of dictionaries containing article data
    """
    # Get the current script's directory
    current_dir = Path(__file__).parent
    # Go up one level to the Integrisme directory and get the data file
    data_file = current_dir.parent / 'integrisme_data.json'
    
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_top_terms(word_frequencies_path, n_terms=50):
    """
    Identify the most important terms using TF-IDF scoring and filtering.
    
    Args:
        word_frequencies_path (str): Path to word frequencies JSON file (currently unused)
        n_terms (int): Number of top terms to return (default: 50)
    
    Returns:
        list: Top N terms sorted by importance score
        
    Notes:
        - Filters out terms that are:
          * Less than 3 characters
          * Contain numbers
          * Non-alphabetic
          * Below 75th percentile in frequency
          * Appear in less than 5% of documents
        - Uses TF-IDF scoring to rank terms by importance
    """
    # Get the current script's directory
    current_dir = Path(__file__).parent
    word_frequencies_file = current_dir.parent / 'Word_cloud/data/word_frequencies.json'
    
    # Load articles data first
    articles = load_integrisme_data()
    
    with open(word_frequencies_file, 'r', encoding='utf-8') as f:
        word_freq = json.load(f)
    
    # First filter: remove problematic terms
    filtered_freq = {
        word: freq for word, freq in word_freq.items() 
        if len(word) > 2  # Filter out very short words
        and not any(c.isdigit() for c in word)  # Filter out words with numbers
        and word not in ['celer', 'célér']  # Explicitly exclude problematic words
        and word.isalpha()  # Only keep purely alphabetic words
        and freq >= np.percentile(list(word_freq.values()), 75)  # Only keep words in top 25% by frequency
    }
    
    # Calculate document frequency (how many articles contain each word)
    doc_frequencies = defaultdict(int)
    for article in articles:
        content = article.get('bibo:content', [{}])[0].get('@value', '').lower()
        for word in filtered_freq:
            if word in content:
                doc_frequencies[word] += 1
    
    # Filter out words that appear in too few documents
    min_doc_frequency = len(articles) * 0.05  # Word should appear in at least 5% of articles
    filtered_freq = {
        word: freq for word, freq in filtered_freq.items()
        if doc_frequencies[word] >= min_doc_frequency
    }
    
    # Calculate TF-IDF scores
    total_articles = len(articles)
    word_importance = {}
    
    for word, freq in filtered_freq.items():
        # Count documents containing this word
        doc_count = sum(1 for article in articles 
                       if word in article.get('bibo:content', [{}])[0].get('@value', '').lower())
        
        # Calculate importance score (TF-IDF)
        importance = freq * np.log(total_articles / (1 + doc_count))
        word_importance[word] = importance
    
    # Get top N terms by importance score
    return list(dict(sorted(word_importance.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:n_terms]).keys())

def calculate_cooccurrence(articles, top_terms, window_type='article'):
    """
    Calculate co-occurrence matrix for the specified terms at different text window levels.
    
    Args:
        articles (list): List of article dictionaries
        top_terms (list): List of terms to analyze
        window_type (str): Level of text analysis - 'article', 'paragraph', or 'sentence'
    
    Returns:
        numpy.ndarray: Co-occurrence matrix where cell [i,j] represents how often
                      term i co-occurs with term j in the specified window type
    
    Notes:
        - Matrix is symmetric (co-occurrence is bidirectional)
        - Applies filtering to remove weak connections:
          * Values below 25th percentile are set to 0
          * Terms with fewer than 3 connections are removed
    """
    n = len(top_terms)
    # Create mapping of terms to matrix indices for efficient lookup
    term_to_idx = {term: i for i, term in enumerate(top_terms)}
    matrix = np.zeros((n, n))
    
    for article in articles:
        content = article.get('bibo:content', [{}])[0].get('@value', '').lower()
        
        # Split content into appropriate windows based on window_type
        if window_type == 'article':
            windows = [content]  # Whole article as one window
        elif window_type == 'paragraph':
            windows = [p.strip() for p in content.split('\n\n') if p.strip()]  # Split by double newline
        elif window_type == 'sentence':
            windows = [s.strip() for s in content.split('.') if s.strip()]  # Split by period
            
        # Process each window
        for window in windows:
            # Find which terms appear in this window
            present_terms = [term for term in top_terms if term in window]
            
            # Update co-occurrence counts
            for i, term1 in enumerate(present_terms):
                for term2 in present_terms[i:]:
                    idx1, idx2 = term_to_idx[term1], term_to_idx[term2]
                    matrix[idx1][idx2] += 1
                    if idx1 != idx2:  # Don't double-count diagonal
                        matrix[idx2][idx1] += 1
    
    # Filter out weak connections
    non_zero_vals = matrix[matrix > 0]
    if len(non_zero_vals) > 0:
        # Keep only values above 25th percentile
        threshold = np.percentile(non_zero_vals, 25)
        matrix[matrix < threshold] = 0
        
        # Remove terms with insufficient connections
        min_connections = 3
        connected_terms = np.sum(matrix > 0, axis=0) >= min_connections
        
        # Zero out rows/columns for terms with too few connections
        for i in range(len(matrix)):
            if not connected_terms[i]:
                matrix[i, :] = 0
                matrix[:, i] = 0

    return matrix

def generate_matrix_data():
    """
    Generate and save co-occurrence network data for visualization.
    
    This function:
    1. Loads the article data
    2. Gets the top terms
    3. Calculates co-occurrence matrices for different window types
    4. Converts matrices to network format (nodes and links)
    5. Saves the result to a JSON file
    
    Output format:
    {
        "window_type": {
            "nodes": [{"id": int, "name": str}, ...],
            "links": [{"source": int, "target": int, "value": float}, ...]
        }
    }
    """
    # Load data
    articles = load_integrisme_data()
    
    # Get top terms from word frequencies
    top_terms = get_top_terms(word_frequencies_path=None, n_terms=30)
    
    # Calculate co-occurrence matrices for different window types
    matrices = {
        'article': calculate_cooccurrence(articles, top_terms, 'article'),
        'paragraph': calculate_cooccurrence(articles, top_terms, 'paragraph'),
        'sentence': calculate_cooccurrence(articles, top_terms, 'sentence')
    }
    
    # Initialize output data structure
    output_data = {window_type: {
        "nodes": [{"id": i, "name": term} for i, term in enumerate(top_terms)],
        "links": []
    } for window_type in matrices.keys()}
    
    # Convert matrices to network links
    n = len(top_terms)
    for window_type, matrix in matrices.items():
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > 0:
                    output_data[window_type]["links"].append({
                        "source": i,
                        "target": j,
                        "value": matrix[i][j]
                    })
    
    # Ensure output directory exists
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    # Save network data to JSON file
    output_file = output_dir / 'cooccurrence.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_matrix_data() 