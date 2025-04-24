import pandas as pd
import numpy as np
from datetime import timedelta
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
news_df = pd.read_csv('analyst_ratings_processed.csv', parse_dates=['date']).dropna()
news_df['date'] = pd.to_datetime(news_df['date'], utc=True).dt.tz_localize(None)

prices_df = pd.read_csv('prices-split-adjusted.csv', parse_dates=['date']).dropna()
prices_df['date'] = pd.to_datetime(prices_df['date'])

# Clean column names
news_df.rename(columns={'title': 'headline', 'stock': 'ticker'}, inplace=True)
prices_df.rename(columns={'symbol': 'ticker'}, inplace=True)

# Filter Fridays and compute Monday/Tuesday opens
friday_prices = prices_df[prices_df['date'].dt.dayofweek == 4][['ticker', 'date', 'close']]
friday_prices.rename(columns={'date': 'friday_date', 'close': 'friday_close'}, inplace=True)
friday_prices['monday'] = friday_prices['friday_date'] + timedelta(days=3)
friday_prices['tuesday'] = friday_prices['friday_date'] + timedelta(days=4)

monday_open = prices_df[['ticker', 'date', 'open']].rename(columns={'date': 'monday', 'open': 'monday_open'})
tuesday_open = prices_df[['ticker', 'date', 'open']].rename(columns={'date': 'tuesday', 'open': 'tuesday_open'})

merged = pd.merge(friday_prices, monday_open, on=['ticker', 'monday'], how='left')
merged = pd.merge(merged, tuesday_open, on=['ticker', 'tuesday'], how='left')

merged['next_open'] = merged['monday_open'].combine_first(merged['tuesday_open'])
merged.dropna(subset=['next_open', 'friday_close'], inplace=True)
merged['target'] = np.where(merged['next_open'] > merged['friday_close'], 'UP', 'DOWN')
merged['open_day'] = np.where(~merged['monday_open'].isna(), 'Monday',
                              np.where(~merged['tuesday_open'].isna(), 'Tuesday', 'Missing'))

# Determine headline window
merged['open_date_used'] = merged['monday']
merged.loc[merged['monday_open'].isna(), 'open_date_used'] = merged['tuesday']
merged['start_date'] = merged['open_date_used'] - timedelta(days=7)
merged['end_date'] = merged['open_date_used'] - timedelta(days=1)
merged = merged.reset_index().rename(columns={'index': 'orig_index'})

# Expand and aggregate news headlines
news_expanded = pd.merge(
    merged[['orig_index', 'ticker', 'start_date', 'end_date']],
    news_df[['ticker', 'date', 'headline']],
    on='ticker', how='left'
)

mask = (news_expanded['date'] >= news_expanded['start_date']) & \
       (news_expanded['date'] <= news_expanded['end_date'])

news_expanded = news_expanded[mask]

headline_agg = news_expanded.groupby('orig_index')['headline'] \
    .apply(lambda x: " ".join(x)).reset_index()

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

merged['weekly_headlines_clean'] = merged['weekly_headlines'].apply(clean)

# Output variables
X = merged['weekly_headlines_clean']
y = merged['target']
