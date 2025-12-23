import os
import requests
import time
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
    This makes the recommender 2x smarter.
    """
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        'api_key': API_KEY,
        'append_to_response': 'credits,keywords' # Get everything in one shot
    }
    
    try:
        r = requests.get(url, params=params)
        if r.status_code != 200: return "", "", []
        
        data = r.json()
        
        # 1. Get Director
        crew = data.get('credits', {}).get('crew', [])
        director = next((person['name'] for person in crew if person['job'] == 'Director'), "")
        
        # 2. Get Top 4 Actors
        cast = data.get('credits', {}).get('cast', [])
        top_cast = [person['name'] for person in cast[:4]]
        
        # 3. Get Top 6 Keywords (e.g., "time travel", "dystopia")
        keywords = [k['name'] for k in data.get('keywords', {}).get('keywords', [])[:6]]
        
        return director, top_cast, keywords
    except:
        return "", [], []

def ingest_high_quality_movies(target_count=50, reset=False):
    print(f"--- 🌟 Starting Super-Ingest (Target: {target_count}, Reset: {reset}) ---")
    
    rec = MovieRecommender()
    existing_ids = set()

    # Setup the brain (Fresh or Load existing)
    if reset or not os.path.exists('models/metadata.pkl'):
        print("⚠️  Mode: RESET. Creating fresh database...")
        rec.index = faiss.IndexFlatL2(384)
    else:
        print("📥 Mode: APPEND. Loading existing database...")
        rec.load('models')
        existing_ids = set(rec.df['id'].tolist())
        print(f"   Found {len(existing_ids)} existing movies.")

    genre_map = get_genre_map()
    movies_added = 0
    page = 1
    
    while movies_added < target_count:
        # Discover movies (The Filter)
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
            results = response.json().get('results', [])
            
            if not results: break

            batch_added = 0
            for m in results:
                if movies_added >= target_count: break
                if m['id'] in existing_ids: continue

                # --- THE UPGRADE IS HERE ---
                # We fetch extra details for every single movie
                director, cast, keywords = get_movie_details(m['id'])
                
                # Convert list of genres to names
                genres = [genre_map.get(gid, '') for gid in m.get('genre_ids', [])]
                
                # Build the SUPER SOUP 🍲
                # We repeat the title twice to give it weight
                # We add Director, Cast, and Keywords into the mix
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
                
                # Sleep a tiny bit to be nice to TMDB API (we are making more calls now)
                time.sleep(0.05)

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
    # Reset=True ensures we rebuild the old movies with the NEW metadata
    ingest_high_quality_movies(target_count=500, reset=True)
