import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# --- Data Loading & Cleaning ---
def safe_date_parse(date_str):
    """Handle multiple date formats gracefully"""
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT

# Load data
news_df = pd.read_csv('analyst_ratings_processed.csv', 
                     parse_dates=['date'],
                     date_parser=safe_date_parse)
prices_df = pd.read_csv('nasdq.csv',
                      parse_dates=['Date'],
                      date_parser=safe_date_parse)

# Clean column names
news_df = news_df.rename(columns={'title': 'headline', 'stock': 'ticker'})
prices_df = prices_df.rename(columns={'Date': 'date', 'Open': 'open', 'Close': 'close'})

# Remove invalid dates
news_df = news_df[~news_df['date'].isna()]
prices_df = prices_df[~prices_df['date'].isna()]

# --- Feature Engineering ---
# Merge datasets
merged_df = pd.merge(news_df, prices_df, on=['ticker', 'date'], how='inner').dropna()

# Find all Mondays with their previous Fridays
mondays = merged_df[merged_df['date'].dt.dayofweek == 0].copy()
mondays['prev_friday'] = mondays['date'] - timedelta(days=3)

# Get Friday closes
friday_prices = merged_df[merged_df['date'].dt.dayofweek == 4][['ticker', 'date', 'close']]
mondays = pd.merge(
    mondays,
    friday_prices.rename(columns={'date': 'prev_friday', 'close': 'friday_close'}),
    on=['ticker', 'prev_friday'],
    how='inner'
)

# Create target variable
mondays['target'] = np.where(mondays['open'] > mondays['friday_close'], 'UP', 'DOWN')

# --- Text Processing (TF-IDF Alternative to BERT) ---
def get_weekly_headlines(ticker, monday_date, news_data):
    """Aggregate all headlines from previous week"""
    start_date = monday_date - timedelta(days=7)
    end_date = monday_date - timedelta(days=1)
    weekly_news = news_data[
        (news_data['ticker'] == ticker) &
        (news_data['date'].between(start_date, end_date))
    ]
    return " ".join(weekly_news['headline'].tolist())

mondays['weekly_headlines'] = mondays.apply(
    lambda row: get_weekly_headlines(row['ticker'], row['date'], merged_df), 
    axis=1
)

# Remove empty headlines
mondays = mondays[mondays['weekly_headlines'].str.strip() != ""]

# Vectorize text using TF-IDF
print("Vectorizing text...")
tfidf = TfidfVectorizer(max_features=1000)  # Limit features for memory
X_text = tfidf.fit_transform(mondays['weekly_headlines'])

# --- Model Training ---
X = X_text
y = mondays['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(clf, 'stock_news_predictor_tfidf.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')