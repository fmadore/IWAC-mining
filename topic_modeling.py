"""
Topic Modeling Script for Article Analysis

This script performs Latent Dirichlet Allocation (LDA) topic modeling on a collection of articles.
It processes pre-lemmatized text data from JSON files and identifies key topics and their distributions
across the document corpus.

Key Features:
- Interactive file selection from data directory
- Processing of pre-lemmatized article texts
- LDA topic modeling with optimized parameters
- Model evaluation metrics for parameter tuning
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
from typing import List, Dict, Tuple, Any, Optional, Union
from scipy.stats import entropy
from sklearn.model_selection import GridSearchCV

# Setup logging configuration for tracking script execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ModelConfig:
    """Configuration parameters for topic modeling."""
    # Topic modeling parameters
    n_topics: int = 10
    n_words_per_topic: int = 10
    max_iter: int = 200
    learning_offset: float = 50.0
    doc_topic_prior: float = 0.1
    topic_word_prior: float = 0.01
    
    # Vectorization parameters
    max_features: Optional[int] = 2000  # None means no limit
    max_df: float = 0.95
    min_df: Union[int, float] = 2  # Can be count or proportion
    min_term_length: int = 3
    max_term_length: int = 30
    term_pattern: str = r'[a-zA-ZÀ-ÿ]+'
    
    # Statistical thresholds for term selection
    use_tfidf_filtering: bool = True
    tfidf_threshold: float = 0.01  # Lowered from 0.1 to be less aggressive
    min_doc_frequency_pct: float = 0.01  # Minimum document frequency as percentage
    max_doc_frequency_pct: float = 0.95  # Maximum document frequency as percentage
    min_features: int = 100  # Minimum number of features to retain after filtering
    
    # Auto-tuning parameters
    auto_tune: bool = False
    topic_range: Tuple[int, int] = (5, 20)
    grid_search_cv: int = 5

class ModelEvaluator:
    """Handles model evaluation metrics for topic modeling."""

    @staticmethod
    def compute_coherence(topic_word_dist: np.ndarray, feature_names: np.ndarray) -> float:
        """
        Compute topic coherence using normalized pointwise mutual information (NPMI).
        Higher values indicate more coherent topics.
        """
        n_top_words = 10
        coherence_scores = []

        for topic in topic_word_dist:
            top_word_indices = topic.argsort()[:-n_top_words-1:-1]
            top_words = feature_names[top_word_indices]
            score = 0
            pairs = 0
            
            for i in range(len(top_words)):
                for j in range(i + 1, len(top_words)):
                    score += ModelEvaluator._npmi(top_words[i], top_words[j])
                    pairs += 1
            
            coherence_scores.append(score / pairs if pairs > 0 else 0)
        
        return np.mean(coherence_scores)

    @staticmethod
    def _npmi(word1: str, word2: str) -> float:
        """Calculate normalized pointwise mutual information between two words."""
        # This is a simplified version; in practice, you'd use actual word co-occurrence statistics
        return 0.0  # Placeholder for actual NPMI calculation

    @staticmethod
    def compute_perplexity(model: LatentDirichletAllocation, doc_term_matrix: Any) -> float:
        """
        Compute perplexity score for the model.
        Lower perplexity indicates better generalization.
        """
        return model.perplexity(doc_term_matrix)

    @staticmethod
    def compute_topic_diversity(topic_word_dist: np.ndarray, n_top_words: int = 10) -> float:
        """
        Compute topic diversity as the proportion of unique words in top N words across all topics.
        Higher diversity indicates more distinct topics.
        """
        top_words = set()
        total_words = 0
        
        for topic in topic_word_dist:
            top_indices = topic.argsort()[:-n_top_words-1:-1]
            top_words.update(top_indices)
            total_words += n_top_words
        
        return len(top_words) / total_words

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

class VectorizerFactory:
    """Factory class for creating and configuring vectorizers with advanced term selection."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._stop_words = self._get_stop_words()

    def _get_stop_words(self) -> List[str]:
        """Get enhanced stop words list for French text."""
        return [
            # Common verbs
            'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir', 'pouvoir',
            'falloir', 'vouloir', 'venir', 'devoir', 'prendre', 'parler', 'mettre',
            
            # Common adverbs and prepositions
            'plus', 'très', 'bien', 'aussi', 'encore', 'puis', 'où', 'or', 'donc',
            'car', 'mais', 'avec', 'sans', 'dans', 'sous', 'sur', 'vers', 'chez',
            
            # Determiners and pronouns
            'tout', 'tous', 'toute', 'toutes', 'autre', 'autres', 'même', 'mêmes',
            'quel', 'quels', 'quelle', 'quelles', 'mon', 'ton', 'son', 'notre',
            
            # Conjunctions and transition words
            'comme', 'ainsi', 'puis', 'ensuite', 'enfin', 'donc', 'car', 'mais',
            'ou', 'et', 'ni', 'soit', 'cependant', 'néanmoins', 'toutefois',
            
            # Common abbreviations and short words
            'm', 'el', 'dr', 'mr', 'mme', 'mm', 'etc', 'cf',
            
            # Numbers written as words
            'un', 'une', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit',
            'neuf', 'dix', 'cent', 'mille'
        ]

    def create_vectorizer(self, texts: List[str] = None) -> CountVectorizer:
        """
        Create and configure a CountVectorizer with advanced term selection criteria.
        
        Args:
            texts: Optional list of texts to analyze for adaptive thresholds
        """
        # Calculate adaptive document frequency thresholds if texts are provided
        if texts and len(texts) > 0:
            min_df = max(2, int(len(texts) * self.config.min_doc_frequency_pct))
            max_df = min(len(texts) - 1, int(len(texts) * self.config.max_doc_frequency_pct))
        else:
            min_df = self.config.min_df
            max_df = self.config.max_df

        # Create token pattern with length constraints
        token_pattern = f'(?u)\\b[a-zA-ZÀ-ÿ]{{{self.config.min_term_length},{self.config.max_term_length}}}\\b'

        return CountVectorizer(
            max_df=max_df,
            min_df=min_df,
            max_features=self.config.max_features,
            stop_words=self._stop_words,
            # Additional parameters for term filtering
            lowercase=True,  # Convert all text to lowercase
            ngram_range=(1, 1),  # Unigrams only for topic modeling
            token_pattern=token_pattern
        )

class TopicModeler:
    """Handles topic modeling operations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.vectorizer_factory = VectorizerFactory(config)
        self.vectorizer = None
        self.lda_model = None
        self.evaluator = ModelEvaluator()
        self.feature_names = None
        self.tfidf_scores = None

    def _create_vectorizer(self, texts: List[str] = None) -> CountVectorizer:
        """Create and configure the CountVectorizer."""
        return self.vectorizer_factory.create_vectorizer(texts)

    def _filter_terms_by_tfidf(self, doc_term_matrix: Any, feature_names: np.ndarray) -> Tuple[Any, np.ndarray]:
        """
        Filter terms based on TF-IDF scores to retain only meaningful terms.
        Includes safeguards to prevent over-filtering.
        
        Returns:
            Tuple of (filtered document-term matrix, filtered feature names)
        """
        from sklearn.feature_extraction.text import TfidfTransformer
        
        if not self.config.use_tfidf_filtering:
            return doc_term_matrix, feature_names

        # Calculate TF-IDF scores
        tfidf = TfidfTransformer()
        tfidf_matrix = tfidf.fit_transform(doc_term_matrix)
        
        # Calculate mean TF-IDF score for each term
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        
        # Sort terms by TF-IDF scores
        term_scores = list(zip(feature_names, mean_tfidf))
        term_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Determine threshold dynamically if needed
        if len(feature_names) > 0:
            # If using threshold would result in too few features, take top N features instead
            threshold = self.config.tfidf_threshold
            selected_terms = mean_tfidf >= threshold
            n_selected = np.sum(selected_terms)
            
            if n_selected < self.config.min_features:
                logging.warning(f"TF-IDF threshold {threshold:.4f} would retain only {n_selected} features. "
                              f"Adjusting to keep top {self.config.min_features} features.")
                # Take top N features instead
                top_n_threshold = term_scores[min(self.config.min_features, len(term_scores)-1)][1]
                selected_terms = mean_tfidf >= top_n_threshold
        else:
            selected_terms = np.ones(len(feature_names), dtype=bool)
        
        self.tfidf_scores = mean_tfidf[selected_terms]
        
        # Filter document-term matrix and feature names
        filtered_matrix = doc_term_matrix[:, selected_terms]
        filtered_features = feature_names[selected_terms]
        
        # Log filtering results
        n_filtered = len(feature_names) - len(filtered_features)
        if n_filtered > 0:
            logging.info(f"Filtered {n_filtered} low-importance terms "
                        f"(retained {len(filtered_features)} terms)")
            logging.info(f"TF-IDF score range of retained terms: "
                        f"{np.min(self.tfidf_scores):.4f} - {np.max(self.tfidf_scores):.4f}")
        
        return filtered_matrix, filtered_features

    def _create_lda_model(self, n_topics: int = None) -> LatentDirichletAllocation:
        """Create and configure the LDA model."""
        if n_topics is None:
            n_topics = self.config.n_topics
            
        return LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='batch',
            max_iter=self.config.max_iter,
            learning_offset=self.config.learning_offset,
            doc_topic_prior=self.config.doc_topic_prior,
            topic_word_prior=self.config.topic_word_prior,
            verbose=1
        )

    def auto_tune_parameters(self, doc_term_matrix: Any) -> Tuple[int, float, float]:
        """
        Automatically tune the number of topics and hyperparameters using grid search.
        Returns optimal n_topics, doc_topic_prior, and topic_word_prior.
        """
        logging.info("Starting automatic parameter tuning...")
        
        # Calculate total number of combinations
        n_topics_range = range(self.config.topic_range[0], self.config.topic_range[1] + 1, 2)
        doc_priors = [0.01, 0.1, 0.5, 1.0]
        word_priors = [0.01, 0.1, 0.5, 1.0]
        total_combinations = len(n_topics_range) * len(doc_priors) * len(word_priors)
        total_fits = total_combinations * self.config.grid_search_cv
        
        logging.info(f"Grid search will evaluate {total_combinations} parameter combinations "
                    f"with {self.config.grid_search_cv}-fold cross-validation "
                    f"(total of {total_fits} model fits)")

        param_grid = {
            'n_components': n_topics_range,
            'doc_topic_prior': doc_priors,
            'topic_word_prior': word_priors
        }

        def custom_scorer(estimator, X):
            """Simple scorer that returns negative perplexity (since GridSearchCV maximizes the score)"""
            return -estimator.perplexity(X)

        base_model = LatentDirichletAllocation(
            learning_method='batch',
            max_iter=self.config.max_iter,
            learning_offset=self.config.learning_offset,
            random_state=42,
            verbose=0
        )
        
        logging.info("Starting grid search (this may take a while)...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.config.grid_search_cv,
            n_jobs=-1,
            verbose=1,  # Use sklearn's built-in verbosity
            scoring=custom_scorer
        )
        
        grid_search.fit(doc_term_matrix)
        
        best_params = grid_search.best_params_
        cv_results = grid_search.cv_results_
        
        # Log detailed results
        logging.info("\nGrid Search Results:")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best perplexity score: {-grid_search.best_score_:.2f}")
        
        # Log top 3 parameter combinations
        mean_scores = -cv_results['mean_test_score']  # Convert back to actual perplexity
        std_scores = cv_results['std_test_score']
        sorted_idx = np.argsort(mean_scores)[:3]
        
        logging.info("\nTop 3 parameter combinations:")
        for idx in sorted_idx:
            params = {k: cv_results[f'param_{k}'][idx] for k in param_grid.keys()}
            logging.info(f"Parameters: {params}")
            logging.info(f"Mean perplexity: {mean_scores[idx]:.2f} (+/- {std_scores[idx] * 2:.2f})")
        
        return (
            best_params['n_components'],
            best_params['doc_topic_prior'],
            best_params['topic_word_prior']
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

    def evaluate_model(self, doc_term_matrix: Any, feature_names: np.ndarray) -> Dict[str, float]:
        """Evaluate the current model using multiple metrics."""
        if self.lda_model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        return {
            'perplexity': self.evaluator.compute_perplexity(self.lda_model, doc_term_matrix),
            'coherence': self.evaluator.compute_coherence(self.lda_model.components_, feature_names),
            'topic_diversity': self.evaluator.compute_topic_diversity(self.lda_model.components_)
        }

    def perform_topic_modeling(self, texts: List[str]) -> Tuple[List[Dict], List[List[float]], Dict[str, float]]:
        """Perform topic modeling on the processed texts."""
        logging.info("Creating document-term matrix...")
        self.vectorizer = self._create_vectorizer(texts)
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Apply TF-IDF filtering if enabled
        doc_term_matrix, feature_names = self._filter_terms_by_tfidf(doc_term_matrix, feature_names)
        self.feature_names = feature_names
        
        logging.info(f"Final vocabulary size: {len(feature_names)} terms")
        
        if self.config.auto_tune:
            logging.info("Auto-tuning model parameters...")
            n_topics, doc_prior, word_prior = self.auto_tune_parameters(doc_term_matrix)
            self.config.n_topics = n_topics
            self.config.doc_topic_prior = doc_prior
            self.config.topic_word_prior = word_prior
        
        logging.info("Creating LDA model with optimized parameters...")
        self.lda_model = self._create_lda_model()
        
        logging.info("Performing LDA topic modeling...")
        with tqdm(total=2, desc="LDA Progress") as pbar:
            self.lda_model.fit(doc_term_matrix)
            pbar.update(1)
            doc_topics = self.lda_model.transform(doc_term_matrix)
            pbar.update(1)
        
        topics = self.extract_topics(feature_names, self.lda_model.components_)
        metrics = self.evaluate_model(doc_term_matrix, feature_names)
        
        # Add vocabulary statistics to metrics
        metrics.update({
            'vocabulary_size': len(feature_names),
            'mean_term_tfidf': float(np.mean(self.tfidf_scores)) if self.tfidf_scores is not None else None,
            'median_term_tfidf': float(np.median(self.tfidf_scores)) if self.tfidf_scores is not None else None
        })
        
        logging.info("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            if value is not None:
                logging.info(f"{metric}: {value:.4f}")
        
        return topics, doc_topics.tolist(), metrics

class TopicModelingPipeline:
    """Orchestrates the complete topic modeling workflow."""
    
    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = self._get_user_config()
        self.config = config
        self.file_handler = FileHandler()
        self.article_processor = ArticleProcessor()
        self.topic_modeler = TopicModeler(config)

    def _get_user_config(self) -> ModelConfig:
        """Get configuration parameters from user input."""
        print("\nTopic Modeling Configuration")
        print("---------------------------")
        auto_tune = input("Would you like to automatically tune the parameters? (y/n): ").lower() == 'y'
        
        if auto_tune:
            min_topics = int(input("Enter minimum number of topics (default: 5): ") or 5)
            max_topics = int(input("Enter maximum number of topics (default: 20): ") or 20)
            return ModelConfig(
                auto_tune=True,
                topic_range=(min_topics, max_topics)
            )
        else:
            n_topics = int(input("Enter number of topics (default: 10): ") or 10)
            return ModelConfig(n_topics=n_topics)

    def prepare_output_data(self, topics: List[Dict], doc_topics: List[List[float]], 
                          titles: List[str], dates: List[str], publishers: List[str],
                          metrics: Dict[str, float]) -> Dict:
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
            ],
            'model_metrics': metrics,
            'model_parameters': {
                'n_topics': self.config.n_topics,
                'doc_topic_prior': self.config.doc_topic_prior,
                'topic_word_prior': self.config.topic_word_prior,
                'auto_tuned': self.config.auto_tune
            }
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
        topics, doc_topics, metrics = self.topic_modeler.perform_topic_modeling(texts)
        
        # Prepare and save results
        output_data = self.prepare_output_data(topics, doc_topics, titles, dates, publishers, metrics)
        self.file_handler.save_results(output_data, input_filename)

def main():
    """Main execution function."""
    pipeline = TopicModelingPipeline()
    pipeline.run()

if __name__ == "__main__":
    main() 