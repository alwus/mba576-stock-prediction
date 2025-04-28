import pandas as pd
import numpy as np
from datetime import timedelta
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

# Load data
news_df = pd.read_csv('analyst_ratings_processed.csv', parse_dates=['date']).dropna()
news_df['date'] = pd.to_datetime(news_df['date'], utc=True).dt.tz_localize(None)

prices_df = pd.read_csv('prices-split-adjusted.csv', parse_dates=['date']).dropna()
prices_df['date'] = pd.to_datetime(prices_df['date'])

# Standardize column names
news_df.rename(columns={'title': 'headline', 'stock': 'ticker'}, inplace=True)
prices_df.rename(columns={'symbol': 'ticker'}, inplace=True)

# Get Friday close prices and corresponding Monday/Tuesday
friday_prices = prices_df[prices_df['date'].dt.dayofweek == 4][['ticker', 'date', 'close']]
friday_prices.rename(columns={'date': 'friday_date', 'close': 'friday_close'}, inplace=True)
friday_prices['monday'] = friday_prices['friday_date'] + timedelta(days=3)
friday_prices['tuesday'] = friday_prices['friday_date'] + timedelta(days=4)

# Get Monday and Tuesday open prices
monday_open = prices_df[['ticker', 'date', 'open']].rename(columns={'date': 'monday', 'open': 'monday_open'})
tuesday_open = prices_df[['ticker', 'date', 'open']].rename(columns={'date': 'tuesday', 'open': 'tuesday_open'})

# Merge open prices
merged = pd.merge(friday_prices, monday_open, on=['ticker', 'monday'], how='left')
merged = pd.merge(merged, tuesday_open, on=['ticker', 'tuesday'], how='left')

# Determine next available open price
merged['next_open'] = merged['monday_open'].combine_first(merged['tuesday_open'])
merged.dropna(subset=['next_open', 'friday_close'], inplace=True)

# Target label: UP or DOWN
merged['target'] = np.where(merged['next_open'] > merged['friday_close'], 'UP', 'DOWN')

# Determine which open day is used
merged['open_date_used'] = merged['monday']
merged.loc[merged['monday_open'].isna(), 'open_date_used'] = merged['tuesday']

# Define headline window
merged['start_date'] = merged['open_date_used'] - timedelta(days=7)
merged['end_date'] = merged['open_date_used'] - timedelta(days=1)
merged = merged.reset_index().rename(columns={'index': 'orig_index'})

# Expand headlines
news_expanded = pd.merge(
    merged[['orig_index', 'ticker', 'start_date', 'end_date']],
    news_df[['ticker', 'date', 'headline']],
    on='ticker', how='left'
)

# Filter headlines within date window
news_expanded = news_expanded[
    (news_expanded['date'] >= news_expanded['start_date']) &
    (news_expanded['date'] <= news_expanded['end_date'])
]

# Aggregate headlines per week/ticker
headline_agg = news_expanded.groupby('orig_index')['headline'] \
    .apply(lambda x: " ".join(x)).reset_index()

# Merge back aggregated headlines
merged = pd.merge(merged, headline_agg, on='orig_index', how='left')
merged.rename(columns={'headline': 'weekly_headlines'}, inplace=True)
merged = merged[merged['weekly_headlines'].str.strip().notna()]
merged.drop(columns=['start_date', 'end_date', 'orig_index'], inplace=True)

# Clean headlines
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    return " ".join(lemma.lemmatize(word) for word in punc_free.split())

merged['headlines'] = merged['weekly_headlines'].apply(clean)
merged['pct_change'] = (merged['next_open'] - merged['friday_close']) / merged['friday_close'] * 100

# Save feature set: X (date, ticker, clean headlines)
X = merged[['open_date_used', 'ticker', 'headlines']]
X.to_csv('X_weekly.csv', index=False)

# Save labels: y (date, ticker, UP/DOWN)
y = merged[['open_date_used', 'ticker', 'target', 'pct_change']]
y.to_csv('y_weekly.csv', index=False)
