import re
from nltk.stem import WordNetLemmatizer

class Preprocessor:
    def __init__(self, stopwords_path):
        with open(stopwords_path, 'r') as file:
            self.stop_words = set(file.read().split())
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text, extra_stopwords=None):
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        stop_words = self.stop_words
        if extra_stopwords:
            stop_words = stop_words.union(extra_stopwords)

        cleaned = [self.lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        return cleaned
