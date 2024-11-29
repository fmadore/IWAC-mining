def generate_comparative_statistics(data: List[Dict]):
    """Generate comparative statistics between raw and lemmatized analysis."""
    raw_sentiments = []
    lemmatized_sentiments = []
    dates = []
    
    for article in data:
        if 'sentiment_analysis' in article and 'date' in article:
            sentiment = article['sentiment_analysis']
            if sentiment['raw'] and sentiment['lemmatized']:
                raw_sentiments.append(sentiment['raw'])
                lemmatized_sentiments.append(sentiment['lemmatized'])
                dates.append(article['date'])
    
    df = pd.DataFrame({
        'date': dates,
        'raw_compound': [s['compound'] for s in raw_sentiments],
        'lemmatized_compound': [s['compound'] for s in lemmatized_sentiments],
        'raw_positive': [s['positive'] for s in raw_sentiments],
        'lemmatized_positive': [s['positive'] for s in lemmatized_sentiments],
        'raw_negative': [s['negative'] for s in raw_sentiments],
        'lemmatized_negative': [s['negative'] for s in lemmatized_sentiments],
    })
    
    # Calculate correlation between raw and lemmatized scores
    correlations = {
        'compound_correlation': float(df['raw_compound'].corr(df['lemmatized_compound'])),
        'positive_correlation': float(df['raw_positive'].corr(df['lemmatized_positive'])),
        'negative_correlation': float(df['raw_negative'].corr(df['lemmatized_negative']))
    }
    
    # Calculate differences in predictions
    differences = {
        'compound_mean_diff': float((df['raw_compound'] - df['lemmatized_compound']).mean()),
        'positive_mean_diff': float((df['raw_positive'] - df['lemmatized_positive']).mean()),
        'negative_mean_diff': float((df['raw_negative'] - df['lemmatized_negative']).mean())
    }
    
    stats_file = os.path.join(os.path.dirname(__file__), 'sentiment_comparison.json')
    statistics = {
        'correlations': correlations,
        'differences': differences,
        'summary': {
            'raw_mean_compound': float(df['raw_compound'].mean()),
            'lemmatized_mean_compound': float(df['lemmatized_compound'].mean()),
            'raw_std_compound': float(df['raw_compound'].std()),
            'lemmatized_std_compound': float(df['lemmatized_compound'].std())
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2) 