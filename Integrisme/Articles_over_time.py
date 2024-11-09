import pandas as pd
import plotly.express as px
from datetime import datetime

# Read the CSV file from GitHub
url = "https://raw.githubusercontent.com/fmadore/Islam-West-Africa-Collection/main/Metadata/CSV/newspaper_articles.csv"
df = pd.read_csv(url)

# Convert date column to datetime, handling both YYYY-MM-DD and YYYY-MM formats
def parse_date(date_str):
    try:
        if pd.isna(date_str):
            return None
        if len(date_str.split('-')) == 2:
            return datetime.strptime(date_str + '-01', '%Y-%m-%d')
        return datetime.strptime(date_str, '%Y-%m-%d')
    except:
        return None

# Convert dates and handle missing values
df['date'] = df['dcterms:date'].apply(parse_date)

# Get the earliest and latest dates
min_date = df['date'].min()
max_date = df['date'].max()

# Filter for articles containing "Intégrisme"
df['has_integrisme'] = df['dcterms:subject'].fillna('').str.contains('Intégrisme', case=False)

# Create a date range covering all articles by year
date_range = pd.date_range(start=min_date, end=max_date, freq='YE')
date_df = pd.DataFrame({'date': date_range})

# Group by date and count articles with "Intégrisme"
yearly_counts = df[df['has_integrisme']].groupby(pd.Grouper(key='date', freq='YE')).size().reset_index()
yearly_counts.columns = ['date', 'count']

# Merge with full date range to include zeros
yearly_counts = pd.merge(date_df, yearly_counts, on='date', how='left')
yearly_counts['count'] = yearly_counts['count'].fillna(0)

# Create the visualization
fig = px.line(yearly_counts, 
              x='date', 
              y='count',
              title='Number of Articles Mentioning "Intégrisme" Over Time',
              labels={'date': 'Year', 'count': 'Number of Articles'},
              template='plotly_white')

# Customize the layout
fig.update_layout(
    showlegend=False,
    hovermode='x unified',
    plot_bgcolor='white',
    xaxis_title='Year',
    yaxis_title='Number of Articles',
    title_x=0.5,
)

# Add markers to the line
fig.update_traces(mode='lines+markers')

# Show the plot
fig.show()

# Save the plot as HTML file
fig.write_html("integrisme_over_time.html")
