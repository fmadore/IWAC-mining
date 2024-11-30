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
    # Create document-term matrix with better French preprocessing
    vectorizer = CountVectorizer(
        max_df=0.95,  # Remove terms that appear in >95% of docs
        min_df=2,     # Remove terms that appear in <2 docs
        stop_words=[  # Add French stop words and common contractions
            'qu', 'être', 'avoir', 'faire', 'pouvoir', 'falloir', 'aller',
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'ce', 'ces',
            'il', 'elle', 'ils', 'elles', 'nous', 'vous', 'leur', 'leurs',
            'qui', 'que', 'quoi', 'dont', 'où', 'quand', 'comment',
            'pourquoi', 'car', 'si', 'tout', 'plus', 'même', 'aussi',
            'autre', 'autres', 'encore', 'toujours', 'alors', 'après',
            'avant', 'bien', 'comme', 'dans', 'pour', 'par', 'sur', 'avec',
            'sans', 'sous', 'entre', 'vers', 'chez', 'donc', 'quand', 'mais',
            'ou', 'et', 'donc', 'or', 'ni', 'car', 'qu'
        ],
        token_pattern=r'[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]+',  # Include French accents
        max_features=1000  # Limit vocabulary size
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Create and fit LDA model with better parameters
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='batch',
        max_iter=100,
        learning_offset=50.,
        doc_topic_prior=1/n_topics,  # Symmetric prior
        topic_word_prior=1/n_topics   # Symmetric prior
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
                       for i in topic.argsort()[:-n_words-1:-1]
                       if len(feature_names[i]) > 2]
        
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