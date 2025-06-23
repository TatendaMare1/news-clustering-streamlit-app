import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter



def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


def cluster_articles(csv_path='news.csv'):
    # Load data
    df = pd.read_csv(csv_path)

    # Preprocess text for keyword analysis (but not for clustering)
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Use section as cluster label
    df['cluster'] = df['section'].astype('category').cat.codes

    # Get unique section names for cluster labels
    section_names = df['section'].unique()
    df['cluster_name'] = df['section']

    # Save clustered data
    df.to_csv('clustered_articles.csv', index=False)

    return df, section_names


def get_top_keywords(df, n=10):
    """Get top keywords for each section cluster"""
    cluster_keywords = {}
    for cluster_name in df['cluster_name'].unique():
        cluster_texts = df[df['cluster_name'] == cluster_name]['processed_text']
        all_words = []
        for text in cluster_texts:
            if isinstance(text, str):
                all_words.extend(text.split())
        word_counts = Counter(all_words)
        cluster_keywords[cluster_name] = word_counts.most_common(n)
    return cluster_keywords
