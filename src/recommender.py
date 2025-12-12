import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

class MovieRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load the Transformer model
        self.encoder = SentenceTransformer(model_name)
        # Dimension of the embedding (384 for MiniLM)
        self.d = 384 
        self.index = None
        self.movies = pd.DataFrame()
        self.id_map = {} # Maps FAISS index ID to Movie ID

    def preprocess_and_embed(self, df):
        self.movies = df.reset_index(drop=True)
        
        # CPU OPTIMIZATION: Process in batches
        # This prevents RAM spikes and shows you a progress bar
        batch_size = 64 
        print(f"Generating embeddings on CPU in batches of {batch_size}...")
        
        # sentence-transformers handles batching internally, 
        # but setting explicit batch_size helps on CPU
        embeddings = self.encoder.encode(
            self.movies['soup'].tolist(), 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print("Normalizing vectors...")
        faiss.normalize_L2(embeddings)
        
        print("Building Index...")
        # IndexFlatIP is brute-force but highly optimized for CPU.
        # It relies on BLAS libraries (MKL/OpenBLAS) which your CPU uses naturally.
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(embeddings)
        
        self.id_map = {i: row['id'] for i, row in self.movies.iterrows()}
        print(f"Done. Indexed {self.index.ntotal} movies.")

    def search(self, query_vector, k=5):
        """
        Low-level search: Input vector -> Top K Movie IDs
        """
        # Ensure vector is 2D (1, d) and normalized
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # FAISS returns -1 if not found
                movie_id = self.id_map[idx]
                results.append({
                    "movie_id": int(movie_id),
                    "score": float(distances[0][i]),
                    "title": self.movies[self.movies['id'] == movie_id]['title'].values[0]
                })
        return results

    def recommend_by_movie(self, movie_title, k=5):
        """
        Find movies similar to a specific movie title already in DB.
        """
        # Find the movie's vector
        movie_row = self.movies[self.movies['title'].str.contains(movie_title, case=False)]
        if movie_row.empty:
            return "Movie not found."
            
        # Re-encode is safer to ensure we have the exact vector logic, 
        # or we could cache the vectors in the dataframe to save time.
        # Here we re-encode for simplicity.
        vec = self.encoder.encode([movie_row.iloc[0]['soup']])
        return self.search(vec, k)
    
    def recommend_on_text(self, text_query, k=5):
        """
        Recommends movies based on a raw text description.
        Example: "A romantic movie about a sinking ship" -> Titanic
        """
        print(f"Searching for: '{text_query}'...")
        
        # 1. Encode the user's text into a vector
        # This runs on the CPU in milliseconds because it's just one sentence.
        query_vector = self.encoder.encode([text_query])
        
        # 2. Normalize (Important for Cosine Similarity)
        faiss.normalize_L2(query_vector)
        
        # 3. Search the existing index
        # We reuse the same search logic we defined earlier
        return self.search(query_vector, k)

    def recommend_for_user(self, liked_movie_titles, k=5):
        """
        The Personalized Logic.
        1. Get vectors for all movies the user liked.
        2. Average them to create a 'User Profile Vector'.
        3. Search against the index.
        """
        vectors = []
        for title in liked_movie_titles:
            # Find movie in our DB
            movie_row = self.movies[self.movies['title'].str.contains(title, case=False)]
            if not movie_row.empty:
                soup = movie_row.iloc[0]['soup']
                vectors.append(self.encoder.encode(soup))
        
        if not vectors:
            return "No known movies provided."

        # Average the vectors (User Profile)
        user_vector = np.mean(vectors, axis=0)
        return self.search(user_vector, k)

    def add_new_movie(self, movie_dict):
        """
        Incremental Learning: Add a movie without retraining.
        movie_dict: {'id': 123, 'title': 'New Movie', 'soup': 'description...'}
        """
        # 1. Vectorize the new soup
        new_vec = self.encoder.encode([movie_dict['soup']])
        faiss.normalize_L2(new_vec)
        
        # 2. Add to FAISS Index (Instant)
        self.index.add(new_vec)
        
        # 3. Update Metadata DataFrame
        # We append the new row to the pandas dataframe
        new_row = pd.DataFrame([movie_dict])
        self.movies = pd.concat([self.movies, new_row], ignore_index=True)
        
        # 4. Update ID Map
        # The new index ID is the last position in the dataframe
        next_idx = len(self.movies) - 1
        self.id_map[next_idx] = movie_dict['id']

    def save(self, path='models/'):
        """Persistence"""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, 'movie_index.faiss'))
        self.movies.to_pickle(os.path.join(path, 'metadata.pkl'))
        # Note: We don't save the encoder, we load it fresh every time.

    def load(self, path='models/'):
        self.index = faiss.read_index(os.path.join(path, 'movie_index.faiss'))
        self.movies = pd.read_pickle(os.path.join(path, 'metadata.pkl'))
        self.id_map = {i: row['id'] for i, row in self.movies.iterrows()}