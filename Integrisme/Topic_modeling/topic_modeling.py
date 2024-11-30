import json
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_processed_data():
    """Load the processed articles from the JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Get parent directory
    data_path = os.path.join(parent_dir, 'integrisme_data.json')
    
    logging.info(f"Looking for data file at: {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}")
        raise

def extract_article_texts(data):
    """Extract processed article texts from the data."""
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
                        texts.append(item['processed_text']['article'])
                        dates.append(article.get('dcterms:date', [{'@value': 'unknown'}])[0]['@value'])
                        titles.append(article.get('o:title', 'Untitled'))
                        publishers.append(publisher)
            elif isinstance(content, dict):
                if '@value' in content and 'processed_text' in content:
                    texts.append(content['processed_text']['article'])
                    dates.append(article.get('dcterms:date', [{'@value': 'unknown'}])[0]['@value'])
                    titles.append(article.get('o:title', 'Untitled'))
                    publishers.append(publisher)
    
    return texts, dates, titles, publishers

def perform_topic_modeling(texts, n_topics=10, n_words=10):
    """Perform LDA topic modeling on the texts."""
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Create and fit LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='batch'
    )
    
    # Fit the model and transform documents
    doc_topics = lda_model.fit_transform(doc_term_matrix)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract top words for each topic
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-n_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'id': topic_idx,
            'words': top_words,
            'weight': float(topic.sum())  # Convert to float for JSON serialization
        })
    
    return topics, doc_topics.tolist()

def main():
    # Load and process data
    data = load_processed_data()
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
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'topic_modeling_results.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Topic modeling results saved to {output_path}")

if __name__ == "__main__":
    main() 