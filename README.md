# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Python and machine learning techniques.  
The system recommends movies similar to a given movie based on metadata such as genres, cast, crew, and overview.

---

## 📌 Project Overview

This project implements a **content-based recommender system** that suggests movies by measuring similarity between movies using textual features.  
It is designed as an end-to-end ML project including data preprocessing, feature engineering, model building, and a simple application interface.

---

## 🧠 How It Works

1. Movie metadata (genres, keywords, cast, crew, overview) is combined into a single text representation.
2. Text is vectorized using **TF-IDF / CountVectorizer**.
3. **Cosine similarity** is used to compute similarity between movies.
4. Given a movie title, the system returns the top similar movies.

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- NLP (text preprocessing & vectorization)  
- Jupyter Notebook  
- Streamlit / Flask (for app interface)

---

## 📂 Project Structure

