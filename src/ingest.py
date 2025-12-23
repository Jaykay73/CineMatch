import os
import requests
import time
from dotenv import load_dotenv
import faiss
from recommender import MovieRecommender

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

def get_genre_map():
    url = f"{BASE_URL}/genre/movie/list"
    params = {'api_key': API_KEY, 'language': 'en-US'}
    try:
        response = requests.get(url, params=params)
        return {g['id']: g['name'] for g in response.json().get('genres', [])}
    except:
        return {}

def ingest_high_quality_movies(target_count=500):
    print(f"--- 🌟 Starting High-Quality Ingest (Rating > 7.0) ---")
    
    # 2. Start Fresh & MANUALLY FIX THE INDEX
    rec = MovieRecommender()
    
   
    rec.index = faiss.IndexFlatL2(384) 
        
    existing_ids = set() 
    
    genre_map = get_genre_map()
    movies_added = 0
    page = 1
    
    while movies_added < target_count:
        # ENDPOINT CHANGE: Use 'discover' to filter server-side
        url = f"{BASE_URL}/discover/movie"
        params = {
            'api_key': API_KEY,
            'language': 'en-US',
            'sort_by': 'popularity.desc', 
            'vote_average.gte': 7.0,      # Only Good Movies
            'vote_count.gte': 500,        # Only Popular-ish Movies
            'page': page
        }

        try:
            response = requests.get(url, params=params)
            results = response.json().get('results', [])
            
            if not results: break

            batch_added = 0
            for m in results:
                if movies_added >= target_count: break
                if m['id'] in existing_ids: continue

                genres = [genre_map.get(gid, '') for gid in m.get('genre_ids', [])]
                soup = f"{m['title']} {m['title']} {' '.join(genres)} {m.get('overview', '')}"

                rec.add_new_movie({
                    'id': m['id'],
                    'title': m['title'],
                    'soup': soup,
                    'rating': m.get('vote_average')
                })
                existing_ids.add(m['id'])
                movies_added += 1
                batch_added += 1

            print(f"Page {page}: Added {batch_added} movies. (Total: {movies_added})")
            
            if movies_added >= target_count: break
            page += 1
            time.sleep(0.2)

        except Exception as e:
            print(f"Error: {e}")
            break

    print("--- Saving New Brain ---")
    rec.save('models/')

if __name__ == "__main__":
    ingest_high_quality_movies(target_count=500)