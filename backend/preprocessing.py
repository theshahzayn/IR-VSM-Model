import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download resources only once
nltk.download('wordnet')
nltk.download('stopwords')


class Preprocessor:
    def __init__(self, stopwords_path):
        # Load custom stopwords from given file
        with open(stopwords_path, 'r') as file:
            self.stop_words = set(file.read().split())
        self.lemmatizer = WordNetLemmatizer()

    # Preprocess Text
    def preprocess(self, text, extra_stopwords=None):
        text = text.lower()  # Case folding
        words = re.findall(r'\b\w+\b', text)  # Tokenize

        # Optional: Combine extra stopwords for queries
        if extra_stopwords:
            stop_words = self.stop_words.union(extra_stopwords)
        else:
            stop_words = self.stop_words

        # Remove stopwords + Lemmatize
        cleaned = [self.lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        return cleaned
