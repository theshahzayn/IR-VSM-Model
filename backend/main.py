from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import os
import numpy as np
import difflib
import re
from math import sqrt
from preprocessing import Preprocessor

app = Flask(__name__)
CORS(app)

ABSTRACTS_FOLDER = "abstracts"
INDEX_FOLDER = "index"
ALPHA = 0.05  # For Normal Search

# Load Index Files
with open(f"{INDEX_FOLDER}/tf_idf_vectors.pkl", "rb") as f:
    doc_vectors = pickle.load(f)

with open(f"{INDEX_FOLDER}/dictionary.json", "r") as f:
    vocab_data = json.load(f)
vocab = vocab_data["dictionary_terms"]

with open(f"{INDEX_FOLDER}/positional_index.json", "r") as f:
    positional_index = json.load(f)

preprocessor = Preprocessor("stopwords.txt")


# Suggestion System
def suggest_words(query, max_suggestions=5):
    query = query.lower().strip()
    if not query:
        return []

    close_matches = difflib.get_close_matches(query, vocab, n=max_suggestions, cutoff=0.75)
    filtered_matches = [word for word in vocab if word.startswith(query)]
    suggestions = list(dict.fromkeys(filtered_matches + close_matches))[:max_suggestions]
    return suggestions


@app.route("/suggest", methods=["GET"])
def suggest():
    query = request.args.get("query", "")
    suggestions = suggest_words(query)
    return jsonify({"suggestions": suggestions})


def preprocess_query(query):
    query = query.lower().strip()
    return preprocessor.preprocess(query)


def cosine_similarity(vec1, vec2):
    all_terms = set(vec1.keys()).union(set(vec2.keys()))
    v1 = np.array([vec1.get(term, 0.0) for term in all_terms])
    v2 = np.array([vec2.get(term, 0.0) for term in all_terms])
    dot = np.dot(v1, v2)
    norm1 = sqrt(np.dot(v1, v1))
    norm2 = sqrt(np.dot(v2, v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", "")
    alpha = float(request.args.get("alpha", ALPHA))
    gold_mode = request.args.get("gold", "false").lower() == "true"

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    # Phrase Query
    if query.startswith('"') and query.endswith('"'):
        phrase = query[1:-1]
        words = preprocess_query(phrase)
        results = phrase_query(words)

    else:
        tokens = preprocess_query(query)

        if not tokens:
            return jsonify({"error": "Query has no valid terms"}), 400

        query_tf = {}
        for term in tokens:
            query_tf[term] = query_tf.get(term, 0) + 1

        N = len(doc_vectors)
        df_data = vocab_data["statistics"]["document_frequency"]

        query_vec = {}
        for term, freq in query_tf.items():
            if term in df_data:
                idf = np.log(N / df_data[term])
                query_vec[term] = freq * idf

        results = []

        for doc_id, doc_vec in doc_vectors.items():
            doc_terms = set(doc_vec.keys())

            # Gold Mode: All query terms must exist
            if gold_mode and not set(tokens).issubset(doc_terms):
                continue

            score = cosine_similarity(query_vec, doc_vec)

            if score >= alpha:
                results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        results = [{"doc_id": doc, "score": round(score, 5)} for doc, score in results]


        snippets = {doc['doc_id']: get_document_snippet(doc['doc_id'], tokens if not query.startswith('"') else words) for doc in results}

    return jsonify({"results": results, "snippets": snippets})


def phrase_query(words):
    if not all(word in positional_index for word in words):
        return []

    possible_docs = set(positional_index[words[0]].keys())
    for word in words[1:]:
        possible_docs &= set(positional_index[word].keys())

    results = []
    for doc in possible_docs:
        positions = [positional_index[word][doc] for word in words]

        for start in positions[0]:
            if all((start + i) in positions[i] for i in range(1, len(words))):
                results.append(doc)
                break

    return results


def get_document_snippet(doc_id, query_terms):
    file_path = os.path.join(ABSTRACTS_FOLDER, f"{doc_id}.txt")
    if not os.path.exists(file_path):
        return "Document not found"

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        for term in query_terms:
            content = content.replace(term, f"<mark>{term}</mark>")

        return content[:300] + "..."
    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/document", methods=["GET"])
def get_full_document():
    doc_id = request.args.get("doc_id")
    if not doc_id:
        return jsonify({"error": "Document ID is required"}), 400

    file_path = os.path.join(ABSTRACTS_FOLDER, f"{doc_id}.txt")
    if not os.path.exists(file_path):
        return jsonify({"error": "Document not found"}), 404

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return jsonify({"doc_id": doc_id, "content": content})
    except Exception as e:
        return jsonify({"error": f"Error reading document: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
