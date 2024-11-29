import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from tqdm import tqdm
import logging
import pandas as pd
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SentimentAnalyzer:
    def __init__(self, model_name: str = 'cmarkea/distilcamembert-base-sentiment'):
        """
        Initialize the sentiment analyzer with a pre-trained French model.
        Specifically tuned for news articles and formal text.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of a single text.
        Returns scores for negative, neutral, and positive sentiments.
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
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
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
            
            # Extract date from dcterms:date structure
            if 'dcterms:date' in article:
                date_info = article['dcterms:date']
                if isinstance(date_info, list) and len(date_info) > 0:
                    # Get the first date entry's @value
                    article['date'] = date_info[0].get('@value', '')
                elif isinstance(date_info, dict):
                    article['date'] = date_info.get('@value', '')
                else:
                    article['date'] = ''
            
        processed_data.append(article)
    
    # Save processed data
    output_file = os.path.join(script_dir, 'integrisme_data_with_sentiment.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # Generate summary statistics
    generate_summary_statistics(processed_data)

def clean_date(date_str: str) -> str:
    """Clean and standardize date strings."""
    if not date_str:
        return None
    
    # Handle month ranges by taking the first date
    if '/' in date_str:
        date_str = date_str.split('/')[0]
    
    # Handle different date formats
    if len(date_str) == 7:  # YYYY-MM format
        return f"{date_str}-01"  # Add first day of month
    elif len(date_str) == 4:  # YYYY format
        return f"{date_str}-01-01"  # Add first day of year
    
    return date_str

def generate_summary_statistics(data: List[Dict]):
    """Generate and save summary statistics of the sentiment analysis."""
    sentiments = []
    dates = []
    
    for article in data:
        if 'sentiment_analysis' in article and 'date' in article:
            sentiments.append(article['sentiment_analysis'])
            cleaned_date = clean_date(article['date'])
            if cleaned_date:
                dates.append(cleaned_date)
            else:
                continue  # Skip articles with invalid dates
    
    df = pd.DataFrame({
        'date': dates,
        'compound': [s['compound'] for s in sentiments],
        'positive': [s['positive'] for s in sentiments],
        'negative': [s['negative'] for s in sentiments],
        'neutral': [s['neutral'] for s in sentiments]
    })
    
    # Convert date strings to datetime and sort
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['date'])  # Remove rows with invalid dates
    df = df.sort_values('date')
    
    # Calculate monthly averages
    monthly_avg = df.set_index('date').resample('ME').mean()
    
    # Calculate yearly averages
    yearly_avg = df.set_index('date').resample('YE').mean()
    
    # Convert timestamps to strings for JSON serialization
    monthly_averages = {k.strftime('%Y-%m'): v for k, v in monthly_avg.to_dict(orient='index').items()}
    yearly_averages = {k.strftime('%Y'): v for k, v in yearly_avg.to_dict(orient='index').items()}
    
    # Save statistics
    stats_file = os.path.join(os.path.dirname(__file__), 'sentiment_statistics.json')
    statistics = {
        'overall_stats': {
            'mean_compound': float(df['compound'].mean()),
            'std_compound': float(df['compound'].std()),
            'median_compound': float(df['compound'].median()),
            'most_positive_date': df.loc[df['compound'].idxmax(), 'date'].strftime('%Y-%m-%d'),
            'most_negative_date': df.loc[df['compound'].idxmin(), 'date'].strftime('%Y-%m-%d'),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'total_articles': len(df)
            }
        },
        'monthly_averages': monthly_averages,
        'yearly_averages': yearly_averages,
        'sentiment_distribution': {
            'positive_ratio': float((df['compound'] > 0).mean()),
            'negative_ratio': float((df['compound'] < 0).mean()),
            'neutral_ratio': float((df['compound'] == 0).mean())
        },
        'topic_sentiment': {
            'overall_tone': 'neutral' if abs(df['compound'].mean()) < 0.1 else 
                          'positive' if df['compound'].mean() > 0 else 'negative',
            'sentiment_stability': float(df['compound'].std()),  # Lower means more consistent tone
            'neutrality_ratio': float(df['neutral'].mean())  # Higher means more neutral coverage
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_integrisme_data() 