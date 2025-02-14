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
- tqdm: For progress tracking
"""

import json
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging
from datetime import datetime
import glob
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

# Setup logging configuration for tracking script execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ModelConfig:
    """Configuration parameters for topic modeling."""
    n_topics: int = 10
    n_words_per_topic: int = 10
    max_features: int = 2000
    max_df: float = 0.95
    min_df: int = 2
    max_iter: int = 200
    learning_offset: float = 50.0
    doc_topic_prior: float = 0.1
    topic_word_prior: float = 0.01

class FileHandler:
    """Handles all file-related operations."""
    
    @staticmethod
    def list_available_files() -> str:
        """List and allow selection of JSON files from the data directory."""
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

    @staticmethod
    def load_json_data(file_path: str) -> Tuple[List[Dict], str]:
        """Load and parse JSON file."""
        logging.info(f"Loading data from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f), os.path.basename(file_path)
        except FileNotFoundError:
            logging.error(f"Data file not found at {file_path}")
            raise

    @staticmethod
    def save_results(output_data: Dict, input_filename: str) -> None:
        """Save topic modeling results to JSON file."""
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(workspace_dir, 'Topic_modeling_data')
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f'topic_modeling_results_{base_name}.json'
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Topic modeling results saved to {output_path}")

class ArticleProcessor:
    """Handles article text and metadata extraction."""
    
    @staticmethod
    def extract_publisher(article: Dict) -> str:
        """Extract publisher information from article metadata."""
        if 'dcterms:publisher' in article:
            pub_info = article['dcterms:publisher']
            if isinstance(pub_info, list) and pub_info:
                if 'display_title' in pub_info[0]:
                    return pub_info[0]['display_title']
                elif '@value' in pub_info[0]:
                    return pub_info[0]['@value']
        return "Unknown"

    @staticmethod
    def process_content(content: Any, article: Dict, publisher: str) -> Optional[Tuple[str, str, str, str]]:
        """Process article content and extract relevant information."""
        if isinstance(content, (list, dict)):
            content_list = content if isinstance(content, list) else [content]
            for item in content_list:
                if isinstance(item, dict) and '@value' in item and 'processed_text' in item:
                    processed_text = item['processed_text']['article']
                    if processed_text:
                        date = article.get('dcterms:date', [{'@value': 'unknown'}])[0]['@value']
                        title = article.get('o:title', 'Untitled')
                        return processed_text, date, title, publisher
        return None

    @staticmethod
    def extract_article_data(data: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Extract processed article texts and metadata."""
        texts, dates, titles, publishers = [], [], [], []
        
        logging.info("Extracting article texts and metadata...")
        for article in tqdm(data, desc="Processing articles"):
            if 'bibo:content' in article:
                publisher = ArticleProcessor.extract_publisher(article)
                result = ArticleProcessor.process_content(article['bibo:content'], article, publisher)
                if result:
                    text, date, title, pub = result
                    texts.append(text)
                    dates.append(date)
                    titles.append(title)
                    publishers.append(pub)
        
        return texts, dates, titles, publishers

class TopicModeler:
    """Handles topic modeling operations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.vectorizer = self._create_vectorizer()
        self.lda_model = self._create_lda_model()

    def _create_vectorizer(self) -> CountVectorizer:
        """Create and configure the CountVectorizer."""
        return CountVectorizer(
            max_df=self.config.max_df,
            min_df=self.config.min_df,
            max_features=self.config.max_features,
            token_pattern=r'[a-zA-ZÀ-ÿ]+',
            stop_words=[
                'être', 'avoir', 'faire',
                'plus', 'très', 'bien',
                'tout', 'tous', 'toute', 'toutes',
                'autre', 'autres',
                'comme', 'ainsi',
                'donc', 'car', 'mais',
                'm', 'el'
            ]
        )

    def _create_lda_model(self) -> LatentDirichletAllocation:
        """Create and configure the LDA model."""
        return LatentDirichletAllocation(
            n_components=self.config.n_topics,
            random_state=42,
            learning_method='batch',
            max_iter=self.config.max_iter,
            learning_offset=self.config.learning_offset,
            doc_topic_prior=self.config.doc_topic_prior,
            topic_word_prior=self.config.topic_word_prior,
            verbose=1
        )

    def extract_topics(self, feature_names: np.ndarray, components: np.ndarray) -> List[Dict]:
        """Extract and format topic information."""
        topics = []
        for topic_idx, topic in enumerate(tqdm(components, desc="Processing topics")):
            word_weights = [(feature_names[i], float(topic[i])) 
                           for i in topic.argsort()[:-self.config.n_words_per_topic-1:-1]]
            
            topic_prevalence = float(topic.sum() / components.sum())
            
            topics.append({
                'id': topic_idx,
                'words': [w for w, _ in word_weights],
                'word_weights': word_weights,
                'weight': float(topic.sum()),
                'prevalence': topic_prevalence,
                'label': f"Topic {topic_idx + 1}"
            })
        
        return sorted(topics, key=lambda x: x['prevalence'], reverse=True)

    def perform_topic_modeling(self, texts: List[str]) -> Tuple[List[Dict], List[List[float]]]:
        """Perform topic modeling on the processed texts."""
        logging.info("Creating document-term matrix...")
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        logging.info("Performing LDA topic modeling...")
        with tqdm(total=2, desc="LDA Progress") as pbar:
            self.lda_model.fit(doc_term_matrix)
            pbar.update(1)
            doc_topics = self.lda_model.transform(doc_term_matrix)
            pbar.update(1)
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = self.extract_topics(feature_names, self.lda_model.components_)
        
        return topics, doc_topics.tolist()

class TopicModelingPipeline:
    """Orchestrates the complete topic modeling workflow."""
    
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.file_handler = FileHandler()
        self.article_processor = ArticleProcessor()
        self.topic_modeler = TopicModeler(config)

    def prepare_output_data(self, topics: List[Dict], doc_topics: List[List[float]], 
                          titles: List[str], dates: List[str], publishers: List[str]) -> Dict:
        """Prepare the final output data structure."""
        return {
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

    def run(self):
        """Execute the complete topic modeling pipeline."""
        print("\n=== Topic Modeling Process ===")
        
        # Load and process data
        file_path = self.file_handler.list_available_files()
        data, input_filename = self.file_handler.load_json_data(file_path)
        
        # Extract article data
        texts, dates, titles, publishers = self.article_processor.extract_article_data(data)
        print(f"\nProcessing {len(texts)} articles...")
        
        # Perform topic modeling
        topics, doc_topics = self.topic_modeler.perform_topic_modeling(texts)
        
        # Prepare and save results
        output_data = self.prepare_output_data(topics, doc_topics, titles, dates, publishers)
        self.file_handler.save_results(output_data, input_filename)

def main():
    """Main execution function."""
    pipeline = TopicModelingPipeline()
    pipeline.run()

if __name__ == "__main__":
    main() 