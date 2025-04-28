import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

# 1) Load & rename
news_df   = pd.read_csv('analyst_ratings_processed.csv',  parse_dates=['date']).dropna()
prices_df = pd.read_csv('prices-split-adjusted.csv',   parse_dates=['date']).dropna()

news_df.rename(columns={'title':'headline','stock':'ticker'}, inplace=True)
prices_df.rename(columns={'symbol':'ticker'},      inplace=True)

# 2) Canonicalize both date columns to datetime64[ns], no tz, date-only
news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
prices_df['date'] = pd.to_datetime(prices_df['date'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()

# Check for problematic rows in `news_df['date']`
if news_df['date'].isnull().any():
    print("Warning: Some rows in 'news_df' have invalid dates.")

# 3) Clean & aggregate headlines per ticker/day
stopsets   = set(stopwords.words('english'))
punc       = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def clean(txt: str) -> str:
    words   = [w for w in txt.lower().split() if w not in stopsets]
    no_punc = ''.join(ch for ch in " ".join(words) if ch not in punc)
    return " ".join(lemmatizer.lemmatize(w) for w in no_punc.split())

news_df['headline_clean'] = news_df['headline'].apply(clean)

daily_head = (
    news_df
    .groupby(['ticker','date'])['headline_clean']
    .apply(' '.join)
    .reset_index()
)

# 4) Prepare prices with next-close
prices_df.sort_values(['ticker','date'], inplace=True)
prices_df['next_close']      = prices_df.groupby('ticker')['close'].shift(-1)
prices_df['close_date_next'] = prices_df.groupby('ticker')['date'].shift(-1)

# 5) Compute target and pct_change
prices_df['target']     = np.where(prices_df['next_close'] > prices_df['close'], 'UP','DOWN')
prices_df['pct_change'] = (prices_df['next_close'] - prices_df['close']) / prices_df['close'] * 100

# 6) Drop rows lacking a next-close
prices_df = prices_df.dropna(subset=['next_close','close_date_next']).copy()

# 7) Merge today's prices with today's aggregated headlines
merged = pd.merge(
    prices_df,
    daily_head,
    on=['ticker','date'],
    how='left'
)

# 8) Fill any missing days with no headlines with empty string
merged['headline_clean'] = merged['headline_clean'].fillna('')

# Remove rows with empty headlines
merged = merged[merged['headline_clean'].str.strip() != '']

# 9) Final X and y
X = (
    merged
    .loc[:, ['close_date_next','ticker','headline_clean']]
    .rename(columns={
        'close_date_next':'date_used',
        'headline_clean':'headlines'
    })
)

y = (
    merged
    .loc[:, ['close_date_next','ticker','target','pct_change']]
    .rename(columns={'close_date_next':'date_used'})
)

# 10) Save
X.to_csv('X_daily.csv', index=False)
y.to_csv('y_daily.csv', index=False)

print(f"Saved {len(X)} rows to X_daily.csv and {len(y)} to y_daily.csv")
