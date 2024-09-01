import pandas as pd
import hazm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FunctionTransformer
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
import pickle

# Function for text preprocessing
def persian_text_preprocessor(text):
    stemmer = hazm.Stemmer()
    stopwords = set(hazm.stopwords_list())
    words = text.split(' ')
    words = [stemmer.stem(word) for word in words if word not in stopwords]
    return ' '.join(words)

# Multithreaded processing function
def preprocess_multithreaded(text_series, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed_texts = list(executor.map(persian_text_preprocessor, text_series))
    return processed_texts

# Named function for use in the pipeline
def apply_preprocessing(x):
    return preprocess_multithreaded(x)

# Load data
df = pd.read_csv('persian_news/train.csv', delimiter='\t')

# Extract features and labels
X = df['content']
y = df['label_id']

# Define the pipeline
pipeline = Pipeline([
    ('preprocessor', FunctionTransformer(apply_preprocessing)),  # Use the named function here
    ('vectorizer', TfidfVectorizer()),  # TF-IDF vectorization
    ('classifier', xgb.XGBClassifier(n_estimators=300, random_state=12))  # XGBoost classifier
])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Train the model
pipeline.fit(x_train, y_train)
