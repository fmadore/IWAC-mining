import pandas as pd
import plotly.express as px
from datetime import datetime
import os

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
    df['has_integrisme'] = df['dcterms:subject'].fillna('').str.contains('Intégrisme', case=False)
    
    # Print validation information
    print("Newspapers in data but not in mapping:")
    print(set(df['dcterms:publisher'].unique()) - set(get_newspaper_country_mapping().keys()))
    print("\nRows with unmapped countries:")
    print(df[df['country'].isna()]['dcterms:publisher'].unique())
    
    return df

def prepare_yearly_counts(df):
    """Prepare yearly counts data for visualization."""
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    
    # Create complete range of years
    all_years = pd.DataFrame([(year, country) 
                             for year in range(min_year, max_year + 1)
                             for country in df['country'].unique() if pd.notna(country)],
                            columns=['year', 'country'])
    
    # Group by year and country
    yearly_counts = df[df['has_integrisme'] & df['country'].notna()].groupby(['year', 'country']).size().reset_index()
    yearly_counts.columns = ['year', 'country', 'count']
    
    # Merge with complete year range
    yearly_counts = pd.merge(all_years, yearly_counts, on=['year', 'country'], how='left')
    yearly_counts['count'] = yearly_counts['count'].fillna(0)
    
    return yearly_counts

def create_visualization(yearly_counts):
    """Create and return the visualization."""
    fig = px.bar(yearly_counts, 
                 x='year',
                 y='count',
                 color='country',
                 title='Number of Articles Mentioning "Intégrisme" by Country Over Time',
                 labels={'year': 'Year', 
                        'count': 'Number of Articles',
                        'country': 'Country'},
                 template='plotly_white',
                 barmode='stack')
    
    fig.update_layout(
        barmode='stack',
        plot_bgcolor='white',
        xaxis_title='Year',
        yaxis_title='Number of Articles',
        title_x=0.5,
        legend_title_text='Country',
        xaxis=dict(
            type='category',
            tickmode='linear',
            dtick=1
        )
    )
    
    return fig

def main():
    """Main function to run the analysis."""
    # URL for the data
    url = "https://raw.githubusercontent.com/fmadore/Islam-West-Africa-Collection/main/Metadata/CSV/newspaper_articles.csv"
    
    # Load and prepare data
    df = load_and_prepare_data(url)
    
    # Prepare yearly counts
    yearly_counts = prepare_yearly_counts(df)
    
    # Create visualization
    fig = create_visualization(yearly_counts)
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the output path in the same directory as the script
    output_path = os.path.join(script_dir, "integrisme_by_country_over_time.html")
    
    # Show and save the plot
    fig.show()
    fig.write_html(output_path)

if __name__ == "__main__":
    main()
