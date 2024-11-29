import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import json
import os
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
from torch.nn.functional import softmax
from typing import Dict, List, Union
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SentimentAnalyzer:
    def __init__(self, model_name: str = 'camembert/camembert-base'):
        """
        Initialize the sentiment analyzer with a CamemBERT model.
        
        Args:
            model_name: The name of the pre-trained model to use
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
        self.model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.model.to(self.device)
        
        # Fine-tune the model if not already fine-tuned
        self._ensure_model_fine_tuned()

    def _ensure_model_fine_tuned(self):
        """Checks if the model needs fine-tuning and performs it if necessary."""
        model_path = 'sentiment_model_state.pt'
        if not os.path.exists(model_path):
            logging.info("Fine-tuned model not found. Starting fine-tuning process...")
            self._fine_tune_model()
            torch.save(self.model.state_dict(), model_path)
        else:
            logging.info("Loading fine-tuned model...")
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of a single text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing sentiment scores
        """
        # Prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = softmax(outputs.logits, dim=1)
            
        # Convert to sentiment scores
        scores = probabilities[0].cpu().numpy()
        return {
            'negative': float(scores[0]),
            'neutral': float(scores[1]),
            'positive': float(scores[2]),
            'compound': float(scores[2] - scores[0])  # Compound score between -1 and 1
        }

def process_integrisme_data():
    """Process the integrisme data and add sentiment analysis."""
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '..', 'integrisme_data.json')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each article
    processed_data = []
    for article in tqdm(data, desc="Processing articles"):
        if 'bibo:content' in article:
            content = article['bibo:content']
            
            # Extract raw text from content
            if isinstance(content, list):
                text = ' '.join(item.get('@value', '') for item in content)
            elif isinstance(content, dict):
                text = content.get('@value', '')
            else:
                text = str(content)
            
            # Analyze sentiment
            sentiment_scores = analyzer.analyze_text(text)
            
            # Add sentiment analysis and date to the article
            article['sentiment_analysis'] = sentiment_scores
            if 'dcterms:date' in article:
                article['date'] = article['dcterms:date'].get('@value', '')
            
        processed_data.append(article)
    
    # Save processed data
    output_file = os.path.join(script_dir, 'integrisme_data_with_sentiment.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # Generate summary statistics
    generate_summary_statistics(processed_data)

def generate_summary_statistics(data: List[Dict]):
    """Generate and save summary statistics of the sentiment analysis."""
    sentiments = []
    dates = []
    
    for article in data:
        if 'sentiment_analysis' in article and 'date' in article:
            sentiments.append(article['sentiment_analysis'])
            dates.append(article['date'])
    
    df = pd.DataFrame({
        'date': dates,
        'compound': [s['compound'] for s in sentiments],
        'positive': [s['positive'] for s in sentiments],
        'negative': [s['negative'] for s in sentiments],
        'neutral': [s['neutral'] for s in sentiments]
    })
    
    # Convert date strings to datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate monthly averages
    monthly_avg = df.set_index('date').resample('M').mean()
    
    # Calculate yearly averages
    yearly_avg = df.set_index('date').resample('Y').mean()
    
    # Save statistics
    stats_file = os.path.join(os.path.dirname(__file__), 'sentiment_statistics.json')
    statistics = {
        'overall_stats': {
            'mean_compound': float(df['compound'].mean()),
            'std_compound': float(df['compound'].std()),
            'median_compound': float(df['compound'].median()),
            'most_positive_date': str(df.loc[df['compound'].idxmax(), 'date']),
            'most_negative_date': str(df.loc[df['compound'].idxmin(), 'date'])
        },
        'monthly_averages': monthly_avg.to_dict(orient='index'),
        'yearly_averages': yearly_avg.to_dict(orient='index'),
        'sentiment_distribution': {
            'positive_ratio': float((df['compound'] > 0).mean()),
            'negative_ratio': float((df['compound'] < 0).mean()),
            'neutral_ratio': float((df['compound'] == 0).mean())
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_integrisme_data() 