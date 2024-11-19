import pandas as pd
from datetime import datetime
import os
import json

def parse_date(date_str):
    """Convert string date to datetime, handling both YYYY-MM-DD and YYYY-MM formats."""
    try:
        if pd.isna(date_str):
            return None
        if len(date_str.split('-')) == 2:
            return datetime.strptime(date_str + '-01', '%Y-%m-%d')
        return datetime.strptime(date_str, '%Y-%m-%d')
    except:
        return None

def get_newspaper_country_mapping():
    """Return dictionary mapping newspapers to their countries."""
    return {
        # Benin newspapers
        '24h au Bénin': 'Benin',
        'Agence Bénin Presse': 'Benin',
        'Banouto': 'Benin',
        'Bénin Intelligent': 'Benin',
        'Boulevard des Infos': 'Benin',
        'Daho-Express': 'Benin',
        'Ehuzu': 'Benin',
        'Fraternité': 'Benin',
        "L'Evénement Précis": 'Benin',
        'La Nation': 'Benin',
        'La Nouvelle Tribune': 'Benin',
        'Le Matinal': 'Benin',
        'Les Pharaons': 'Benin',
        'Matin Libre': 'Benin',
        
        # Burkina Faso newspapers
        'Burkina 24': 'Burkina Faso',
        'Carrefour africain': 'Burkina Faso',
        'FasoZine': 'Burkina Faso',
        "L'Evénement": 'Burkina Faso',
        "L'Observateur": 'Burkina Faso',
        "L'Observateur Paalga": 'Burkina Faso',
        'La Preuve': 'Burkina Faso',
        'Le Pays': 'Burkina Faso',
        'LeFaso.net': 'Burkina Faso',
        'Mutations': 'Burkina Faso',
        'San Finna': 'Burkina Faso',
        'Sidwaya': 'Burkina Faso',
        
        # Côte d'Ivoire newspapers
        'Agence Ivoirienne de Presse': 'Côte d\'Ivoire',
        'Alif': 'Côte d\'Ivoire',
        'Fraternité Hebdo': 'Côte d\'Ivoire',
        'Fraternité Matin': 'Côte d\'Ivoire',
        'Ivoire Dimanche': 'Côte d\'Ivoire',
        "L'Intelligent d'Abidjan": 'Côte d\'Ivoire',
        'La Voie': 'Côte d\'Ivoire',
        'Le Jour': 'Côte d\'Ivoire',
        'Le Jour Plus': 'Côte d\'Ivoire',
        'Le Nouvel Horizon': 'Côte d\'Ivoire',
        'Le Patriote': 'Côte d\'Ivoire',
        'Notre Temps': 'Côte d\'Ivoire',
        'Notre Voie': 'Côte d\'Ivoire',
        
        # Togo newspapers
        'Agence Togolaise de Presse': 'Togo',
        'Courrier du Golfe': 'Togo',
        'La Nouvelle Marche': 'Togo',
        'Togo-Presse': 'Togo'
    }

def load_and_prepare_data(url):
    """Load data from URL and prepare it for analysis."""
    df = pd.read_csv(url)
    
    # Convert dates and add country information
    df['date'] = df['dcterms:date'].apply(parse_date)
    df['year'] = df['date'].dt.year
    df['country'] = df['dcterms:publisher'].map(get_newspaper_country_mapping())
    
    # Check for multiple keywords
    keywords = ['Intégrisme', 'Fondamentalisme islamique', 'Islamisme', 'Radicalisation', 
               'Extrémisme', 'Obscurantisme', 'Terrorisme', 'Djihadisme', 'Salafisme']
    for keyword in keywords:
        df[f'has_{keyword.lower().replace(" ", "_")}'] = df['dcterms:subject'].fillna('').str.contains(keyword, case=False)
    
    # Print validation information
    print("Newspapers in data but not in mapping:")
    print(set(df['dcterms:publisher'].unique()) - set(get_newspaper_country_mapping().keys()))
    print("\nRows with unmapped countries:")
    print(df[df['country'].isna()]['dcterms:publisher'].unique())
    
    return df

def prepare_combined_yearly_counts(df):
    """Prepare combined yearly counts data for visualization."""
    # Define the range of years
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    keywords = ['Intégrisme', 'Fondamentalisme islamique', 'Islamisme', 
                'Radicalisation', 'Extrémisme', 'Obscurantisme', 'Terrorisme', 'Djihadisme', 'Salafisme']
    
    # Initialize list to store results
    results = []
    
    # For each year and keyword combination
    for year in range(min_year, max_year + 1):
        year_data = {}
        year_data['year'] = year
        year_data['keywords'] = []
        
        for keyword in keywords:
            # Filter data for this year and keyword
            mask = (df['year'] == year) & (df[f'has_{keyword.lower().replace(" ", "_")}'])
            articles = df[mask]
            
            if len(articles) > 0:
                # Group by country and count
                country_counts = articles.groupby('country').size().to_dict()
                
                keyword_data = {
                    'keyword': keyword,
                    'total_count': len(articles),
                    'countries': [
                        {'name': country, 'count': count}
                        for country, count in country_counts.items()
                    ]
                }
                year_data['keywords'].append(keyword_data)
            else:
                # Include zero counts
                year_data['keywords'].append({
                    'keyword': keyword,
                    'total_count': 0,
                    'countries': []
                })
        
        results.append(year_data)
    
    return results

def main():
    """Main function to run the analysis."""
    # URL for the data
    url = "https://raw.githubusercontent.com/fmadore/Islam-West-Africa-Collection/main/Metadata/CSV/newspaper_articles.csv"
    
    # Load and prepare data
    df = load_and_prepare_data(url)
    
    # Prepare combined yearly counts
    yearly_counts = prepare_combined_yearly_counts(df)
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save the data to a JSON file in the same directory as the script
    json_output_path = os.path.join(script_dir, 'yearly_counts.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(yearly_counts, f, ensure_ascii=False, indent=2)
    
    # Inform the user that the data has been saved
    print(f"Data has been saved to '{json_output_path}' for D3.js visualization.")

if __name__ == "__main__":
    main() 