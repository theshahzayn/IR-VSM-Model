![image](https://github.com/user-attachments/assets/a8e99165-f380-4825-ab2f-bc9e11f9ce38)

# IR-VSM Model 

This project implements a **Vector Space Model (VSM)** based Information Retrieval system using TF-IDF weighting and cosine similarity. It includes support for phrase queries, ranked results, and a clean web interface powered by Flask (backend) and React (frontend).

---

## 📁 Dataset

- A set of 448 research paper abstracts (`abstracts/` folder)
- Custom stopword list (`stopwords.txt`)

---

## ⚙️ Features

✅ TF-IDF Vectorization  
✅ Cosine Similarity-Based Ranking  
✅ Phrase Query Support (using Positional Index)  
✅ Stopword Removal, Lemmatization  
✅ Query Suggestions via Fuzzy Matching  
✅ Flask REST API  
✅ React Frontend 
✅ Snippets with Highlighted Terms  

---

## 🧠 Technologies Used

- Python 3.11
- Flask
- ReactJS
- NLTK
- NumPy
- Pickle / JSON

---

## 🚀 How to Run

### Backend

```bash
cd backend
pip install -r requirements.txt
python indexer.py  # builds all indexes
python main.py     # runs the Flask server on port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
```

---

## 📌 API Endpoints

### `GET /search?query=your+text[&alpha=0.05][&gold=true]`
Returns ranked documents and snippets.  
Supports:
- Normal Queries
- Gold Mode (`&gold=true`)
- Phrase Queries: wrap query in quotes → `"deep learning"`

### `GET /suggest?query=term`
Returns autocomplete suggestions.

### `GET /document?doc_id=123`
Returns full abstract content.

---

## 🧪 Sample Query Usage

```http
/search?query=deep learning
/search?query="human interaction"
/search?query=deep&alpha=0.001&gold=true
```

---

## 📅 Submission Details

- **Course**: Information Retrieval (CS4051)  
- **Assignment**: Programming Assignment 2 — VSM Model  
- **Due Date**: April 15, 2025  
- **Student**: Shahzain Zaidi  
- **University**: FAST-NUCES Karachi  
