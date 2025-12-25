import os
import requests
import time
import pandas as pd
import faiss
from dotenv import load_dotenv
from src.recommender import MovieRecommender

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

def get_genre_map():
    """Fetch genre list once to avoid repeated calls."""
    url = f"{BASE_URL}/genre/movie/list"
    params = {'api_key': API_KEY, 'language': 'en-US'}
    try:
        response = requests.get(url, params=params)
        return {g['id']: g['name'] for g in response.json().get('genres', [])}
    except:
        return {}

def get_movie_details(movie_id):
    """
    Fetches the 'Secret Sauce': Cast, Director, and Keywords.
    """
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        'api_key': API_KEY,
        'append_to_response': 'credits,keywords'
    }
    
    try:
        r = requests.get(url, params=params)
        if r.status_code != 200: return "", "", []
        
        data = r.json()
        
        # 1. Director
        crew = data.get('credits', {}).get('crew', [])
        director = next((person['name'] for person in crew if person['job'] == 'Director'), "")
        
        # 2. Cast (Top 4)
        cast = data.get('credits', {}).get('cast', [])
        top_cast = [person['name'] for person in cast[:4]]
        
        # 3. Keywords (Top 6)
        keywords = [k['name'] for k in data.get('keywords', {}).get('keywords', [])[:6]]
        
        return director, top_cast, keywords
    except:
        return "", [], []

def ingest_high_quality_movies(target_count=500, reset=True):
    print(f"--- 🌟 Starting Super-Ingest (Target: {target_count}, Reset: {reset}) ---")
    
    rec = MovieRecommender()
    existing_ids = set()

    # --- SAFETY FIX: Robust Loading Logic ---
    if reset or not os.path.exists('models/metadata.pkl'):
        print("⚠️  Mode: RESET. Creating fresh database...")
        rec.index = faiss.IndexFlatL2(384)
    else:
        print("📥 Mode: APPEND. Loading existing database...")
        try:
            rec.load('models')
            # Check if 'df' exists and is not empty before reading
            if not rec.df.empty:
                existing_ids = set(rec.df['id'].tolist())
                print(f"   Found {len(existing_ids)} existing movies.")
            else:
                print("   ⚠️ Database loaded but appears empty.")
        except Exception as e:
            print(f"   ❌ Error loading existing DB: {e}. Starting fresh.")
            rec.index = faiss.IndexFlatL2(384)
            rec.df = pd.DataFrame() # Reset dataframe

    genre_map = get_genre_map()
    movies_added = 0
    page = 1
    
    while movies_added < target_count:
        # Discover movies
        url = f"{BASE_URL}/discover/movie"
        params = {
            'api_key': API_KEY,
            'language': 'en-US',
            'sort_by': 'popularity.desc', 
            'vote_average.gte': 7.0,      
            'vote_count.gte': 500,        
            'page': page
        }

        try:
            response = requests.get(url, params=params)
            
            # --- DEBUG FIX: Catch API Errors ---
            if response.status_code != 200:
                print(f"❌ CRITICAL API ERROR: {response.status_code}")
                print(f"Server Message: {response.text}")
                break

            results = response.json().get('results', [])
            
            if not results: 
                print("⚠️ API returned 200 OK, but 'results' list is empty.")
                break

            batch_added = 0
            for m in results:
                if movies_added >= target_count: break
                if m['id'] in existing_ids: continue

                # Fetch details
                director, cast, keywords = get_movie_details(m['id'])
                
                # Genres
                genres = [genre_map.get(gid, '') for gid in m.get('genre_ids', [])]
                
                # Build Soup
                soup = (
                    f"{m['title']} {m['title']} "
                    f"Director: {director} "
                    f"Cast: {' '.join(cast)} "
                    f"Keywords: {' '.join(keywords)} "
                    f"Genres: {' '.join(genres)} "
                    f"{m.get('overview', '')}"
                )

                rec.add_new_movie({
                    'id': m['id'],
                    'title': m['title'],
                    'soup': soup,
                    'rating': m.get('vote_average')
                })
                existing_ids.add(m['id'])
                movies_added += 1
                batch_added += 1
                
                time.sleep(0.05) # Be nice to API

            print(f"Page {page}: Added {batch_added} movies. (Total New: {movies_added})")
            
            if page > 50 and movies_added == 0: break
            if movies_added >= target_count: break
            page += 1

        except Exception as e:
            print(f"Error: {e}")
            break

    print(f"--- Saving Super-Brain (Total Size: {len(existing_ids)}) ---")
    rec.save('models/')

if __name__ == "__main__":
    # Check if we are running inside GitHub Actions
    is_github_action = os.getenv("GITHUB_ACTIONS") == "true"

    if is_github_action:
        print("🤖 AUTOMATION DETECTED: Running Daily 800-Movie Reset.")
        # Daily Refresh: Get top 800, Wipe old data (Reset=True)
        ingest_high_quality_movies(target_count=800, reset=True)
    else:
        print("👨‍💻 LOCAL DEV DETECTED: Running Safe Test.")
        # Local Test: Just add 50 to check if it works, don't delete DB (Reset=False)
        ingest_high_quality_movies(target_count=50, reset=False)