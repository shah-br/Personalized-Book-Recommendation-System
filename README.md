# 📚 Personalized Book Recommender System

A high-performance, scalable book recommender system built from scratch using collaborative filtering and sparse matrix operations. Designed to predict top 5 book suggestions per user based on historical ratings and user similarity—no black-box libraries. Just clean, efficient math and code.

---

## 🔍 Problem Statement

Build a recommender system that intelligently suggests books to users based on:
- Their personal preferences
- Ratings from top-K similar users

The system should be explainable, scalable, and capable of handling sparse, high-dimensional data.

---

## 🧠 Why This Project Stands Out

- ✅ **Fully custom collaborative filtering** (no prebuilt ML libraries)
- ✅ **Efficient sparse matrix handling** using `libsvm` and `scipy`
- ✅ **Top-K cosine similarity algorithm** from scratch
- ✅ **Readable recommendations** with ISBN, title, and relevance score
- ✅ **Scalable design** for real-world datasets

---

## 📁 Dataset Overview
Kaggle Dataset Link - https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset   
Utilized 3 core CSV files:
- `Books.csv`: ISBN, Title, Author, Year, Publisher
- `Users.csv`: UserID, Age
- `Ratings.csv`: UserID, ISBN, Rating

### 👤 Users

- **Total users:** `278,859`
- **Users who rated at least 1 book:** `99,053`
- **Users who rated at least 2 books:** `43,385`
- **Users who rated at least 10 books:** `12,306`

➡️ Many users rate only once or twice — sparse data problem.

### 📚 Books

- **Total unique books:** `271,379`
- **Books with at least 1 rating:** `270,171`
- **Books with at least 2 ratings:** `124,513`
- **Books with at least 10 ratings:** `17,480`

➡️ Long-tail distribution — most books are rated only a few times.

### 🧾 Ratings

- **Rating scale:** `0–10`
- **Matrix sparsity:** Over 99% of the user-book matrix is empty

➡️ Highlights the need for efficient sparse matrix computation and cold-start strategies.

---

## 🧼 Step 1: Data Cleaning & Sparse Matrix Creation

- Filled missing `Age`, `Author`, and `Publisher`
- Removed duplicates
- Mapped `User-ID` and `ISBN` to numerical indices
- Created a COO-format User-Book matrix
- Exported to `libsvm` for compactness and fast parsing

📄 Output: `output.libsvm`

---

## 🧪 Step 2: Collaborative Filtering Engine

### 🔗 Algorithm Overview

1. **Top-K User Similarity**
   - Compute cosine similarity between each user and others
   - Select top-K most similar users

2. **Generate Predictions**
   - For books not yet rated by a user, compute estimated ratings
   - Use similarity-weighted average of neighbor ratings

3. **Recommend**
   - Select top 5 books with highest estimated rating

📄 Output: `final_recommendations_from_libsvm.csv`

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|-------------|
| Language | Python |
| Libraries | `pandas`, `numpy`, `scipy` |
| Format | `libsvm`, `coo_matrix`, `csv` |
| Algorithm | User-based Collaborative Filtering (Cosine Similarity) |

---

## 🚀 How to Run

```bash
# Clone this repo
git clone https://github.com/<your-username>/book-recommender.git
cd book-recommender

# Ensure your CSV files are in the folder

# Step 1: Preprocess data and create libsvm format
python step1_data_prep.py

# Step 2: Run collaborative filtering recommender
python step2_recommender.py
