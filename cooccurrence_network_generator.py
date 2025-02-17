"""
Script to generate co-occurrence matrices and network data for term relationships in articles.

This script analyzes the relationships between terms in a corpus of articles by:
1. Loading article data from JSON (user-selected file)
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
import os
from typing import Union, List

def list_available_datasets():
    """
    List all available JSON datasets in the data directory.
    
    Returns:
        list: List of available JSON files
        Path: Path to the data directory
    """
    # Get the root directory (where the script is located)
    root_dir = Path(__file__).parent
    data_dir = root_dir / 'data'
    
    # Get all JSON files in the data directory
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    return json_files, data_dir

def select_dataset():
    """
    Present available datasets to the user and get their selection.
    
    Returns:
        Union[Path, List[Path]]: Either a single file path or list of all file paths
    """
    json_files, data_dir = list_available_datasets()
    
    if not json_files:
        raise FileNotFoundError("No JSON files found in the data directory")
    
    print("\nAvailable datasets:")
    print("0. Process ALL files")
    for idx, file in enumerate(json_files, 1):
        print(f"{idx}. {file}")
    
    while True:
        try:
            selection = int(input("\nSelect a dataset by number (0 for all): "))
            if selection == 0:
                return [data_dir / file for file in json_files]
            elif 1 <= selection <= len(json_files):
                selected_file = json_files[selection - 1]
                return data_dir / selected_file
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def load_article_data(data_file):
    """
    Load the article data from a JSON file.
    
    Args:
        data_file (Path): Path to the JSON data file
    
    Returns:
        list: List of dictionaries containing article data
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_top_terms(articles, word_frequencies_path=None, n_terms=50):
    """
    Identify the most important terms using TF-IDF scoring and filtering.
    
    Args:
        articles (list): List of article dictionaries
        word_frequencies_path (str): Path to word frequencies JSON file (unused)
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
    # Calculate word frequencies directly from articles
    word_freq = defaultdict(int)
    for article in articles:
        content = article.get('bibo:content', [{}])[0].get('processed_text', {}).get('article', '').lower()
        # Split content into words and count frequencies
        words = content.split()
        for word in words:
            if (len(word) > 2  # Filter out very short words
                and not any(c.isdigit() for c in word)  # Filter out words with numbers
                and word not in ['celer', 'célér']  # Explicitly exclude problematic words
                and word.isalpha()):  # Only keep purely alphabetic words
                word_freq[word] += 1
    
    # First filter: keep only words in top 25% by frequency
    if word_freq:
        freq_threshold = np.percentile(list(word_freq.values()), 75)
        filtered_freq = {
            word: freq for word, freq in word_freq.items() 
            if freq >= freq_threshold
        }
    else:
        print("Warning: No valid words found in the articles")
        return []
    
    # Calculate document frequency (how many articles contain each word)
    doc_frequencies = defaultdict(int)
    for article in articles:
        content = article.get('bibo:content', [{}])[0].get('processed_text', {}).get('article', '').lower()
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
        doc_count = doc_frequencies[word]
        
        # Calculate importance score (TF-IDF)
        importance = freq * np.log(total_articles / (1 + doc_count))
        word_importance[word] = importance
    
    # Get top N terms by importance score
    sorted_terms = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Print some statistics
    print(f"\nWord statistics:")
    print(f"Total unique words found: {len(word_freq)}")
    print(f"Words after frequency filtering: {len(filtered_freq)}")
    print(f"Final number of terms selected: {min(n_terms, len(sorted_terms))}")
    
    return list(dict(sorted_terms[:n_terms]).keys())

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
        # Get the processed text structure from the article
        processed_text = article.get('bibo:content', [{}])[0].get('processed_text', {})
        
        # Get the appropriate windows based on window_type
        if window_type == 'article':
            windows = [processed_text.get('article', '')]
        elif window_type == 'paragraph':
            windows = processed_text.get('paragraphs', [])
        elif window_type == 'sentence':
            windows = processed_text.get('sentences', [])
            
        # Process each window
        for window in windows:
            # Find which terms appear in this window
            present_terms = [term for term in top_terms if term in window.lower()]
            
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
    1. Prompts user to select input data file(s)
    2. Gets the top terms
    3. Calculates co-occurrence matrices for different window types
    4. Converts matrices to network format (nodes and links)
    5. Saves the result to JSON file(s) named 'cooccurrence_[dataset_name].json'
    
    Output format:
    {
        "window_type": {
            "nodes": [{"id": int, "name": str}, ...],
            "links": [{"source": int, "target": int, "value": float}, ...]
        }
    }
    """
    try:
        # Let user select the dataset(s)
        data_files = select_dataset()
        
        # Convert to list if single file
        if not isinstance(data_files, list):
            data_files = [data_files]
            
        # Process each file
        for data_file in data_files:
            dataset_name = data_file.stem  # Get filename without extension
            print(f"\nProcessing dataset: {dataset_name}")
            
            # Load data
            articles = load_article_data(data_file)
            print(f"Loaded {len(articles)} articles")
            
            # Get top terms from word frequencies
            top_terms = get_top_terms(articles, n_terms=30)
            print(f"Identified {len(top_terms)} top terms")
            
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
            
            # Use a subdirectory 'cooccurrence' inside the data directory
            output_dir = data_file.parent / 'cooccurrence'
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f'cooccurrence_{dataset_name}.json'

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"Results saved to: {output_file}")
        
        print("\nAll processing complete!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    generate_matrix_data() 