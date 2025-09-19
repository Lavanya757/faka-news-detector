# src/train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from preprocess import clean_text

# load
df = pd.read_csv("data/sample_fake_news.csv")
df['clean'] = df['text'].apply(clean_text)
X = df['clean']
y = df['label'].map({'real':0,'fake':1})  # binary: 0 real, 1 fake

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# build pipeline (TF-IDF + classifier). Try both models as alternatives.
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_df=0.95)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# save
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/fake_news_pipeline.pkl")
print("Model saved -> models/fake_news_pipeline.pkl")

