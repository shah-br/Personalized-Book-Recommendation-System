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

## 📁 Dataset

Utilized 3 core CSV files:
- `Books.csv`: ISBN, Title, Author, Publisher
- `Users.csv`: User IDs and demographics
- `Ratings.csv`: Ratings given by users to books

---

## 🧼 Step 1: Data Cleaning & Sparse Matrix Creation

- Missing `Age`, `Author`, and `Publisher` fields handled
- Duplicates removed
- `User-ID` and `ISBN` mapped to numeric indices
- Constructed a **COO-format User-Book matrix**
- Exported to `libsvm` format for efficient parsing

📄 Output: `output.libsvm`

---

## 🧪 Step 2: Collaborative Filtering Engine

### Algorithm Breakdown

1. **Top-K Similar Users**
   - Cosine similarity between each user pair
   - Retain top-K (e.g., 10) most similar neighbors

2. **Prediction & Recommendation**
   - For each unread book, estimate score using neighbors' ratings
   - Recommend top 5 books with highest predicted score

📄 Output: `final_recommendations_from_libsvm.csv`

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|-------------|
| Language | Python |
| Libraries | `pandas`, `numpy`, `scipy` |
| Format | `libsvm`, `coo_matrix` |
| Algorithm | User-based Collaborative Filtering (Cosine Similarity) |

---

## 🚀 How to Run

```bash
# Clone this repo
git clone https://github.com/<your-username>/book-recommender.git
cd book-recommender

# Ensure your CSV files are in the same folder

# Run data prep (creates sparse matrix and libsvm file)
python step1_data_prep.py

# Run recommender system
python step2_recommender.py
