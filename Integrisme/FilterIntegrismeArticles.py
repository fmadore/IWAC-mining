import pandas as pd
import os

def load_data(url):
    """Load data from URL."""
    return pd.read_csv(url)

def filter_integrisme_articles(df):
    """Filter articles mentioning 'intégrisme'."""
    return df[df['dcterms:subject'].fillna('').str.contains('Intégrisme', case=False)]

def save_filtered_data(df, file_path):
    """Save the filtered data to a CSV file."""
    df.to_csv(file_path, index=False)

def main():
    """Main function to filter and save articles."""
    # URL for the data
    url = "https://raw.githubusercontent.com/fmadore/Islam-West-Africa-Collection/main/Metadata/CSV/newspaper_articles.csv"
    
    # Load data
    df = load_data(url)
    
    # Filter articles mentioning "intégrisme"
    integrisme_articles = filter_integrisme_articles(df)
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the file path in the same directory
    file_path = os.path.join(script_dir, 'filtered_integrisme_articles.csv')
    
    # Save the filtered data to a new CSV file in the same folder
    save_filtered_data(integrisme_articles, file_path)

if __name__ == "__main__":
    main() 