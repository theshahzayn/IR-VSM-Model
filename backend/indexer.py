import os
import re
import json
import pickle
from collections import defaultdict, Counter
from math import log
from preprocessing import Preprocessor


class Indexer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

        self.inverted_index = defaultdict(set)
        self.positional_index = defaultdict(lambda: defaultdict(list))
        self.tf_idf_vectors = {}
        self.raw_tf = {}
        self.df = defaultdict(int)
        self.dictionary = set()
        self.doc_count = 0

    def index_documents(self, abstracts_dir, stopwords):
        for filename in sorted(os.listdir(abstracts_dir)):
            filepath = os.path.join(abstracts_dir, filename)
            try:
                doc_id = int(os.path.splitext(filename)[0])
            except ValueError:
                doc_id = filename

            self.doc_count += 1
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                words = self.preprocessor.preprocess(file.read(), stopwords)

                tf = Counter(words)
                self.raw_tf[doc_id] = tf

                for pos, word in enumerate(words):
                    self.dictionary.add(word)
                    self.df[word] += 1
                    self.inverted_index[word].add(doc_id)
                    self.positional_index[word][doc_id].append(pos)

        self.compute_tf_idf()

    def compute_tf_idf(self):
        N = self.doc_count
        for doc_id, tf in self.raw_tf.items():
            tf_idf = {}
            for term, freq in tf.items():
                idf = log(N / self.df[term])
                tf_idf[term] = freq * idf
            self.tf_idf_vectors[doc_id] = tf_idf

    def save_all(self, folder="index"):
        os.makedirs(folder, exist_ok=True)

        with open(f"{folder}/inverted_index.json", 'w') as f:
            json.dump({k: list(v) for k, v in self.inverted_index.items()}, f, indent=4)

        with open(f"{folder}/positional_index.json", 'w') as f:
            json.dump({k: {str(d): p for d, p in v.items()} for k, v in self.positional_index.items()}, f, indent=4)

        with open(f"{folder}/dictionary.json", 'w') as f:
            stats = {
                'total_unique_terms': len(self.dictionary),
                'total_documents': self.doc_count,
                'document_frequency': dict(self.df)
            }
            json.dump({'dictionary_terms': list(self.dictionary), 'statistics': stats}, f, indent=4)

        with open(f"{folder}/tf_idf_vectors.pkl", 'wb') as f:
            pickle.dump(self.tf_idf_vectors, f)


# --------------------------- #
# MAIN FUNCTION TO RUN INDEXER
# --------------------------- #

if __name__ == "__main__":
    stopwords_path = "stopwords.txt"
    abstracts_dir = "abstracts/"

    print("Indexing in Progress...")

    preprocessor = Preprocessor(stopwords_path)
    indexer = Indexer(preprocessor)

    indexer.index_documents(abstracts_dir, preprocessor.stop_words)
    indexer.save_all()

    print("All Indexes Created Successfully & Saved!")
