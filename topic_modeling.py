"""
Topic Modeling Script for Article Analysis

This script performs Latent Dirichlet Allocation (LDA) topic modeling on a collection of articles.
It processes pre-lemmatized text data from JSON files and identifies key topics and their distributions
across the document corpus.

Key Features:
- Interactive file selection from data directory
- Processing of pre-lemmatized article texts
- LDA topic modeling with optimized parameters
- Extraction of topic-document and word-topic distributions
- JSON output of modeling results with topic weights and document metadata

Dependencies:
- sklearn: For LDA implementation and text vectorization
- numpy: For numerical operations
- json: For data I/O
- logging: For operation logging
"""

import json
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging
from datetime import datetime
import glob

# Setup logging configuration for tracking script execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def list_available_files():
    """
    List and allow selection of JSON files from the data directory.
    
    Returns:
        str: Full path to the selected JSON file
        
    Raises:
        FileNotFoundError: If no JSON files are found in the data directory
    """
    # Get the workspace root directory (now the same as script directory)
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(workspace_dir, 'data')
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    
    if not json_files:
        logging.error("No JSON files found in the data directory")
        raise FileNotFoundError("No JSON files available")
    
    print("\nAvailable files:")
    for idx, file_path in enumerate(json_files, 1):
        file_name = os.path.basename(file_path)
        print(f"{idx}. {file_name}")
    
    while True:
        try:
            choice = int(input("\nEnter the number of the file you want to process: "))
            if 1 <= choice <= len(json_files):
                return json_files[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def load_processed_data():
    """
    Load and parse the selected JSON file containing article data.
    
    Returns:
        tuple: (data, filename)
            - data: Parsed JSON content containing article information
            - filename: Base name of the selected file
            
    Raises:
        FileNotFoundError: If the selected file cannot be found
        json.JSONDecodeError: If the file contains invalid JSON
    """
    file_path = list_available_files()
    logging.info(f"Loading data from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), os.path.basename(file_path)
    except FileNotFoundError:
        logging.error(f"Data file not found at {file_path}")
        raise

def extract_article_texts(data):
    """
    Extract processed article texts and metadata from the JSON data structure.
    
    This function handles various JSON structure formats and extracts:
    - Lemmatized article text (pre-processed by spaCy)
    - Publication dates
    - Article titles
    - Publisher information
    
    Args:
        data (list): List of article dictionaries from the JSON file
        
    Returns:
        tuple: (texts, dates, titles, publishers)
            - texts: List of processed article texts
            - dates: List of publication dates
            - titles: List of article titles
            - publishers: List of publisher names
    """
    texts = []
    dates = []
    titles = []
    publishers = []
    
    for article in data:
        if 'bibo:content' in article:
            content = article['bibo:content']
            # Extract publisher from dcterms:publisher display_title
            publisher = "Unknown"
            if 'dcterms:publisher' in article:
                pub_info = article['dcterms:publisher']
                if isinstance(pub_info, list) and pub_info:
                    # Get the first publisher's display_title
                    if 'display_title' in pub_info[0]:
                        publisher = pub_info[0]['display_title']
                    elif '@value' in pub_info[0]:
                        publisher = pub_info[0]['@value']
            
            if isinstance(content, list):
                for item in content:
                    if '@value' in item and 'processed_text' in item:
                        # Use the lemmatized article text
                        processed_text = item['processed_text']['article']
                        if processed_text:  # Only add if we have processed text
                            texts.append(processed_text)
                            dates.append(article.get('dcterms:date', [{'@value': 'unknown'}])[0]['@value'])
                            titles.append(article.get('o:title', 'Untitled'))
                            publishers.append(publisher)
            elif isinstance(content, dict):
                if '@value' in content and 'processed_text' in content:
                    # Use the lemmatized article text
                    processed_text = content['processed_text']['article']
                    if processed_text:  # Only add if we have processed text
                        texts.append(processed_text)
                        dates.append(article.get('dcterms:date', [{'@value': 'unknown'}])[0]['@value'])
                        titles.append(article.get('o:title', 'Untitled'))
                        publishers.append(publisher)
    
    return texts, dates, titles, publishers

def perform_topic_modeling(texts, n_topics=10, n_words=10):
    """
    Perform LDA topic modeling on the processed article texts.
    
    The function implements several optimizations:
    - Custom stop words for French text
    - Optimized LDA parameters for better topic separation
    - Prevalence calculation for topic importance
    
    Args:
        texts (list): List of pre-processed article texts
        n_topics (int, optional): Number of topics to extract. Defaults to 10.
        n_words (int, optional): Number of top words per topic. Defaults to 10.
        
    Returns:
        tuple: (topics, doc_topics)
            - topics: List of topic dictionaries containing:
                * id: Topic identifier
                * words: Top words in the topic
                * word_weights: Weights for each top word
                * weight: Overall topic weight
                * prevalence: Topic prevalence in corpus
                * label: Topic label
            - doc_topics: Document-topic distribution matrix
    """
    # Create document-term matrix
    # Note: Many stop words and preprocessing steps are already handled by spaCy
    vectorizer = CountVectorizer(
        max_df=0.95,     # Remove terms that appear in >95% of docs
        min_df=2,        # Remove terms that appear in <2 docs
        max_features=2000,  # Increased from 1000 to capture more nuanced topics
        token_pattern=r'[a-zA-ZÀ-ÿ]+',  # Simplified pattern since text is pre-processed
        # Minimal stop words since most are handled by spaCy
        stop_words=[
            'être', 'avoir', 'faire',  # Common verbs
            'plus', 'très', 'bien',    # Common adverbs
            'tout', 'tous', 'toute', 'toutes',  # Variations of "all"
            'autre', 'autres',         # Variations of "other"
            'comme', 'ainsi',          # Comparison words
            'donc', 'car', 'mais'      # Conjunctions
        ]
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Create and fit LDA model with optimized parameters
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='batch',
        max_iter=200,          # Increased from 100 for better convergence
        learning_offset=50.,
        doc_topic_prior=0.1,   # Adjusted from 1/n_topics for sparser topic distribution
        topic_word_prior=0.01  # Adjusted from 1/n_topics for clearer word-topic associations
    )
    
    # Fit the model and transform documents
    doc_topics = lda_model.fit_transform(doc_term_matrix)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract top words for each topic with their weights
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        # Get word indices and weights sorted by importance
        word_weights = [(feature_names[i], float(topic[i])) 
                       for i in topic.argsort()[:-n_words-1:-1]]
        
        # Calculate topic prevalence (percentage of corpus)
        topic_prevalence = float(topic.sum() / lda_model.components_.sum())
        
        topics.append({
            'id': topic_idx,
            'words': [w for w, _ in word_weights],
            'word_weights': word_weights,
            'weight': float(topic.sum()),
            'prevalence': topic_prevalence,
            'label': f"Topic {topic_idx + 1}"  # We can manually label these later
        })
    
    # Sort topics by prevalence
    topics.sort(key=lambda x: x['prevalence'], reverse=True)
    return topics, doc_topics.tolist()

def main():
    """
    Main execution function that orchestrates the topic modeling process:
    1. Loads and processes the selected JSON file
    2. Extracts article texts and metadata
    3. Performs LDA topic modeling
    4. Saves results to a JSON file with the input filename as base
    
    The output JSON contains:
    - Topic information including top words and weights
    - Document metadata and their topic distributions
    """
    # Load and process data
    data, input_filename = load_processed_data()
    texts, dates, titles, publishers = extract_article_texts(data)
    
    # Perform topic modeling
    topics, doc_topics = perform_topic_modeling(texts)
    
    # Prepare output data
    output_data = {
        'topics': topics,
        'documents': [
            {
                'id': idx,
                'title': title,
                'date': date,
                'publisher': publisher,
                'topic_weights': weights
            }
            for idx, (title, date, publisher, weights) in enumerate(zip(titles, dates, publishers, doc_topics))
        ]
    }
    
    # Create Topic_modeling_data directory if it doesn't exist
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(workspace_dir, 'Topic_modeling_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results in Topic_modeling_data folder
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f'topic_modeling_results_{base_name}.json'
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Topic modeling results saved to {output_path}")

if __name__ == "__main__":
    main() 