import pandas as pd
import numpy as np
import hazm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Custom transformer for text preprocessing
class PersianTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = hazm.Stemmer()
        self.stopwords = hazm.stopwords_list()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(self._process)

    def _process(self, text):
        words = text.split(' ')
        words = [self.stemmer.stem(word) for word in words if word not in self.stopwords]
        return ' '.join(words)

# Load data
df = pd.read_csv('persian_news/train.csv', delimiter='\t')

# Extract features and labels
X = df['content']
y = df['label_id']

# Define the pipeline
pipeline = Pipeline([
    ('preprocessor', PersianTextPreprocessor()),  # Custom text preprocessing
    ('vectorizer', TfidfVectorizer()),  # TF-IDF vectorization
    ('classifier', xgb.XGBClassifier(n_estimators=300, random_state=12))  # XGBoost classifier
])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Train the model
pipeline.fit(x_train, y_train)

import pickle

# Save the pipeline to a file
with open('text_classification_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)
