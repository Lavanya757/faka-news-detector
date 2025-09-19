# src/preprocess.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOP = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)   # remove urls
    text = re.sub(r'\S+@\S+', '', text)                   # remove emails
    text = re.sub(r'[^a-z\s]', ' ', text)                 # keep letters only
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)
