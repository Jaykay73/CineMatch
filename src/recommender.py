import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

class MovieRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # 1. Load the Brain
        self.encoder = SentenceTransformer(model_name)
        self.d = 384 # Dimension for MiniLM
        
        # 2. Initialize Memory (Safety First)
        # We initialize them as empty so the code doesn't crash if accessed before loading
        self.index = None
        self.df = pd.DataFrame() # Replaces 'self.movies'

    def save(self, path='models/'):
        """Saves the index and metadata to disk."""
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save FAISS Index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, 'movie_index.faiss'))
        
        # Save Metadata DataFrame
        with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(self.df, f)

    def load(self, path='models/'):
        """Loads the brain from disk."""
        index_path = os.path.join(path, 'movie_index.faiss')
        meta_path = os.path.join(path, 'metadata.pkl')

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.df = pickle.load(f)

    def add_new_movie(self, movie_data):
        """Adds a single movie to the memory (used during ingest)."""
        # 1. Vectorize
        vector = self.encoder.encode([movie_data['soup']])
        faiss.normalize_L2(vector)
        
        # 2. Add to Index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.d)
        self.index.add(vector)
        
        # 3. Add to DataFrame
        new_row = pd.DataFrame([movie_data])
        if self.df.empty:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], ignore_index=True)

    def get_banned_genres(self, query_text):
        """
        Returns a list of genres to BAN based on the user's vibe.
        """
        query_lower = query_text.lower()
        banned_genres = set() # Use a set to prevent duplicates

        # 1. HAPPY / COMEDY MODE
        if any(w in query_lower for w in ["happy", "uplifting", "comedy", "laugh", "cheerful", "funny"]):
            banned_genres.update(["Horror", "Thriller", "War", "Crime", "Tragedy"])

        # 2. FAMILY / KIDS MODE
        # The trigger "family" is dangerous because of "Crime Family", so we check context.
        if any(w in query_lower for w in ["kid", "child", "animation", "disney", "pixar"]):
            banned_genres.update(["Horror", "Crime", "War", "Romance", "Adult"])
        
        # Special check for "Family" to avoid the "Crime Family" bug
        # We only trigger Family mode if "Crime" is NOT in the query.
        elif "family" in query_lower and "crime" not in query_lower:
             banned_genres.update(["Horror", "Crime", "War", "Romance", "Adult"])

        # 3. ROMANCE MODE
        if "romantic" in query_lower or "romance" in query_lower:
            banned_genres.update(["Horror"])

        # --- 🛡️ THE IMMUNITY RULE 🛡️ ---
        # If the user's query explicitly mentions a genre (e.g. "Crime Drama..."),
        # then that genre is IMMUNE. We must REMOVE it from the ban list.
        
        final_bans = []
        for ban in banned_genres:
            # If the banned genre is actually IN the query text, allow it.
            if ban.lower() in query_lower:
                continue 
            final_bans.append(ban)

        return final_bans
        
    def recommend(self, text_query, k=10):
        """
        Smart Recommendation with Guardrails
        """
        print(f"🔎 Searching for: '{text_query}'")
        
        if self.df.empty or self.index is None:
            return []

        # 1. Get user vector
        query_vector = self.encoder.encode([text_query])
        faiss.normalize_L2(query_vector)

        # 2. OVER-FETCH: Ask for 20 candidates (so we have spares if we delete some)
        distances, indices = self.index.search(query_vector, k=25)
        
        # 3. IDENTIFY BANS
        banned_genres = self.get_banned_genres(text_query)
        if banned_genres:
            print(f"🛡️  Guardrails Active! Banning: {banned_genres}")

        results = []
        seen_titles = set()
        
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.df): continue
            
            # --- SAFE ACCESS ---
            movie_data = self.df.iloc[idx].to_dict()
            
            # --- 4. FILTER LOGIC ---
            movie_soup = movie_data.get('soup', '').lower()
            
            is_banned = False
            for ban in banned_genres:
                if ban.lower() in movie_soup:
                    print(f"🚫 Blocking '{movie_data['title']}' (Contains {ban})")
                    is_banned = True
                    break
            
            if is_banned: continue
            # -----------------------

            # Deduplication
            if movie_data['title'] in seen_titles: continue
            
            results.append({
                'id': int(movie_data['id']),
                'title': movie_data['title'],
                'score': float(distances[0][i]),
            })
            seen_titles.add(movie_data['title'])

            if len(results) >= k:
                break
        
        return results

    def recommend_on_text(self, text_query, k=10):
        """Wrapper for the main recommend function."""
        return self.recommend(text_query, k)

    def recommend_for_user(self, liked_movie_titles, k=10):
        """Personalized Logic based on liked movies."""
        if self.df.empty: return []

        vectors = []
        for title in liked_movie_titles:
            # Search in self.df
            movie_row = self.df[self.df['title'].str.contains(title, case=False, na=False)]
            if not movie_row.empty:
                soup = movie_row.iloc[0]['soup']
                vectors.append(self.encoder.encode(soup))
        
        if not vectors:
            return []

        # Average the vectors
        user_vector = np.mean(vectors, axis=0)
        
        # Search using the user vector (reuse search logic manually here)
        user_vector = user_vector.reshape(1, -1)
        faiss.normalize_L2(user_vector)
        
        distances, indices = self.index.search(user_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.df):
                movie_data = self.df.iloc[idx]
                results.append({
                    'id': int(movie_data['id']),
                    'title': movie_data['title'],
                    'score': float(distances[0][i])
                })
        return results
