Code:
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from scipy.sparse.linalg import norm
# Parsing the already generated output.libsvm file in Task1
def parse_libsvm(file_path):
    rows, cols, data = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            user_id = int(parts[0])
            for book_rating in parts[1:]:
                book_index, rating = map(float, book_rating.split(':'))
                rows.append(user_id)
                cols.append(int(book_index) - 1)  # Convert 1-based to 0-based
                data.append(rating)
    return rows, cols, data

filepath = ''
rows, cols, data = parse_libsvm("output.libsvm")

# Now Reconstructing the sparse matrix
num_users = max(rows) + 1  # Total number of users
num_books = max(cols) + 1  # Total number of books
sparse_matrix = coo_matrix((data, (rows, cols)), shape=(num_users, num_books)).tocsr()

# Loading ISBN-to-Title Mapping
books_df = pd.read_csv("Books.csv", sep=";")  # Replace with your actual Books.csv file path
isbn_to_title = dict(zip(books_df["ISBN"], books_df["Title"]))

# Load or Recreate Index-to-ISBN Mapping. 
book_map = {isbn: idx for idx, isbn in enumerate(books_df["ISBN"].unique())}
index_to_isbn = {v: k for k, v in book_map.items()}  # Reverse mapping

# Calculating Cosine Similarities
def calculate_top_k_similar_users(sparse_matrix, K):
    num_users = sparse_matrix.shape[0]
    top_k_indices = np.zeros((num_users, K), dtype=int)
    top_k_similarities = np.zeros((num_users, K), dtype=float)
    row_norms = norm(sparse_matrix, axis=1)  # Norms for normalization

    for user_id in range(num_users):
        similarity = sparse_matrix[user_id].dot(sparse_matrix.T).toarray().flatten()
        similarity /= (row_norms[user_id] * row_norms + 1e-10)  # Normalize cosine similarity
        top_k = np.argsort(-similarity)[:K]  # Get top-K similar users
        top_k_indices[user_id] = top_k
        top_k_similarities[user_id] = similarity[top_k]

    return top_k_indices, top_k_similarities

K = 10
top_k_indices, top_k_similarities = calculate_top_k_similar_users(sparse_matrix, K)


# Generating Recommendations
def recommend_top_5_books_from_libsvm(
    sparse_matrix, K, top_k_indices, top_k_similarities, index_to_isbn, isbn_to_title
):
    recommendations = []

    for user_id in range(sparse_matrix.shape[0]):
        similar_users = top_k_indices[user_id]
        similarities = top_k_similarities[user_id]

        # Find books read by similar users
        BK = sparse_matrix[similar_users].sum(axis=0).nonzero()[1]

        # Estimate ratings for books not yet read
        book_scores = {}
        for book_id in BK:
            if sparse_matrix[user_id, book_id] == 0:  # User hasn't read this book
                numerator = 0
                denominator = 0
                for idx, similar_user in enumerate(similar_users):
                    rating = sparse_matrix[similar_user, book_id]
                    if rating > 0:
                        numerator += rating * similarities[idx]
                        denominator += similarities[idx]
                if denominator > 0:
                    book_scores[book_id] = numerator / denominator

        # Get top-5 books
        top_books = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for book_id, score in top_books:
            isbn = index_to_isbn.get(book_id, None)
            if isbn:
                title = isbn_to_title.get(isbn, "Unknown Title")
                recommendations.append({
                    "User_ID": user_id,
                    "Book_ID": book_id,
                    "ISBN": isbn,
                    "Book_Title": title,
                    "Recommendation_Score": score
                })

    return pd.DataFrame(recommendations)



recommendations_df = recommend_top_5_books_from_libsvm(
    sparse_matrix=sparse_matrix,
    K=10,
    top_k_indices=top_k_indices,
    top_k_similarities=top_k_similarities,
    index_to_isbn=index_to_isbn,
    isbn_to_title=isbn_to_title
)

# Saving Recommendations into a csv file
recommendations_df.to_csv("final_recommendations_from_libsvm.csv", index=False)
print("Recommendations saved to final_recommendations_from_libsvm.csv")
