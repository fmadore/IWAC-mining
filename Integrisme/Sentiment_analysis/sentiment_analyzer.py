# sentiment_analyzer.py
# This script performs sentiment analysis on French text data using a pre-trained transformer model.
# It processes articles from a JSON file, analyzes their sentiment, and generates statistical summaries.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from tqdm import tqdm
import logging
import pandas as pd
from typing import Dict, List

# Configure logging to track script execution and potential issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SentimentAnalyzer:
    """
    A class for analyzing sentiment in French text using a pre-trained transformer model.
    
    This analyzer uses the DistilCamemBERT model specifically fine-tuned for sentiment analysis
    of French text. It provides sentiment scores across three categories (negative, neutral, positive)
    and a compound score for overall sentiment.
    """
    
    def __init__(self, model_name: str = 'cmarkea/distilcamembert-base-sentiment'):
        """
        Initialize the sentiment analyzer with a pre-trained French model.
        
        Args:
            model_name (str): The name/path of the pre-trained model to use.
                            Defaults to DistilCamemBERT model fine-tuned for sentiment.
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
        Analyze the sentiment of a single text passage.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            Dict[str, float]: A dictionary containing sentiment scores:
                - negative: probability of negative sentiment
                - neutral: probability of neutral sentiment
                - positive: probability of positive sentiment
                - compound: aggregated score between -1 (most negative) and 1 (most positive)
        """
        # Tokenize input text with padding and truncation for transformer model
        # Max length of 512 tokens is standard for BERT-based models
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Perform inference without gradient calculation for efficiency
        with torch.no_grad():
            # Get model predictions
            outputs = self.model(**inputs)
            # Convert logits to probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        # Extract probability scores and convert to numpy for easier handling
        scores = probabilities[0].cpu().numpy()
        
        # Return dictionary with sentiment scores and compound score
        # Compound score is calculated as positive - negative for a single value summary
        return {
            'negative': float(scores[0]),
            'neutral': float(scores[1]),
            'positive': float(scores[2]),
            'compound': float(scores[2] - scores[0])  # Compound score between -1 and 1
        }

def process_integrisme_data():
    """
    Process the integrisme dataset by adding sentiment analysis to each article.
    
    This function:
    1. Loads articles from a JSON file
    2. Analyzes the sentiment of each article's content
    3. Extracts and standardizes publication dates
    4. Saves the enriched data to a new JSON file
    5. Generates statistical summaries of the sentiment analysis
    """
    # Initialize the sentiment analyzer with default model
    analyzer = SentimentAnalyzer()
    
    # Construct path to input file relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '..', 'integrisme_data.json')
    
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each article with progress bar
    processed_data = []
    for article in tqdm(data, desc="Processing articles"):
        # Only process articles with content
        if 'bibo:content' in article:
            content = article['bibo:content']
            
            # Handle different content formats (list or dict)
            if isinstance(content, list):
                # Join multiple content items with spaces
                text = ' '.join(item.get('@value', '') for item in content)
            elif isinstance(content, dict):
                # Extract single content value
                text = content.get('@value', '')
            else:
                # Fallback for unexpected content format
                text = str(content)
            
            # Perform sentiment analysis
            sentiment_scores = analyzer.analyze_text(text)
            article['sentiment_analysis'] = sentiment_scores
            
            # Extract and standardize publication date
            if 'dcterms:date' in article:
                date_info = article['dcterms:date']
                if isinstance(date_info, list) and len(date_info) > 0:
                    # Get first date from list of dates
                    article['date'] = date_info[0].get('@value', '')
                elif isinstance(date_info, dict):
                    # Get date from dictionary format
                    article['date'] = date_info.get('@value', '')
                else:
                    article['date'] = ''
            
        processed_data.append(article)
    
    # Save processed data to new JSON file
    output_file = os.path.join(script_dir, 'integrisme_data_with_sentiment.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # Generate statistical analysis of the results
    generate_summary_statistics(processed_data)

def clean_date(date_str: str) -> str:
    """
    Clean and standardize date strings to ensure consistent format.
    
    Args:
        date_str (str): The input date string to clean
        
    Returns:
        str: A standardized date string in YYYY-MM-DD format, or None if invalid
        
    Examples:
        >>> clean_date("2023-05")
        "2023-05-01"
        >>> clean_date("2023")
        "2023-01-01"
    """
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
    """
    Generate comprehensive statistical summaries of the sentiment analysis results.
    
    This function calculates and saves various statistics including:
    - Overall sentiment metrics (mean, median, std dev)
    - Monthly and yearly sentiment averages
    - Sentiment distribution ratios
    - Topic sentiment analysis
    - Date range statistics
    
    Args:
        data (List[Dict]): List of processed articles with sentiment scores
        
    Output:
        Saves a JSON file containing:
        - Overall statistics
        - Time-based averages
        - Sentiment distribution metrics
        - Topic sentiment analysis
    """
    # Extract sentiment scores and dates from processed data
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
    
    # Create DataFrame for easier statistical analysis
    df = pd.DataFrame({
        'date': dates,
        'compound': [s['compound'] for s in sentiments],
        'positive': [s['positive'] for s in sentiments],
        'negative': [s['negative'] for s in sentiments],
        'neutral': [s['neutral'] for s in sentiments]
    })
    
    # Convert and clean date data
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['date'])  # Remove rows with invalid dates
    df = df.sort_values('date')
    
    # Calculate time-based averages
    monthly_avg = df.set_index('date').resample('ME').mean()  # ME = Month End
    yearly_avg = df.set_index('date').resample('YE').mean()   # YE = Year End
    
    # Convert datetime index to strings for JSON serialization
    monthly_averages = {k.strftime('%Y-%m'): v for k, v in monthly_avg.to_dict(orient='index').items()}
    yearly_averages = {k.strftime('%Y'): v for k, v in yearly_avg.to_dict(orient='index').items()}
    
    # Compile comprehensive statistics
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
            # Determine overall tone based on mean compound score
            'overall_tone': 'neutral' if abs(df['compound'].mean()) < 0.1 else 
                          'positive' if df['compound'].mean() > 0 else 'negative',
            'sentiment_stability': float(df['compound'].std()),  # Lower means more consistent tone
            'neutrality_ratio': float(df['neutral'].mean())  # Higher means more neutral coverage
        }
    }
    
    # Save statistics to JSON file
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Entry point of the script
    # Processes the integrisme data and generates sentiment analysis
    process_integrisme_data() 