from dotenv import load_dotenv
import os
import requests
import pandas as pd
from recommender import MovieRecommender

load_dotenv()

# CONFIGURATION
API_KEY = os.getenv('API_KEY')
BASE_URL = "https://api.themoviedb.org/3"

def get_genre_map():
    """
    Fetches the official list of Genre IDs -> Names.
    Example: {28: 'Action', 12: 'Adventure'}
    """
    url = f"{BASE_URL}/genre/movie/list"
    params = {'api_key': API_KEY, 'language': 'en-US'}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        # Turn list of dicts into a single lookup dictionary
        mapping = {g['id']: g['name'] for g in data['genres']}
        print("Genre map loaded.")
        return mapping
    except Exception as e:
        print(f"Failed to load genres: {e}")
        return {}

def get_popular_movies(limit=100):
    """
    Fetches the current most popular movies.
    """
    movies = []
    pages = (limit // 20) + 1
    
    print(f"Fetching top {limit} popular movies...")
    
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/movie/popular"
        params = {
            'api_key': API_KEY, 
            'language': 'en-US', 
            'page': page
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                results = response.json().get('results', [])
                movies.extend(results)
            else:
                print(f"Error on page {page}: {response.status_code}")
        except Exception as e:
            print(f"Connection failed: {e}")
        
        if len(movies) >= limit:
            break
            
    return movies[:limit]

def run_weekly_update():
    # 1. Load the Brain
    print("Loading existing database...")
    rec = MovieRecommender()
    try:
        rec.load('models/')
        existing_ids = set(rec.movies['id'].values)
        print(f"Database currently holds {len(existing_ids)} movies.")
    except:
        print("No existing model found. Cannot update empty model.")
        return

    # 2. Get Data
    candidates = get_popular_movies(limit=1000)
    genre_map = get_genre_map() # <--- NEW STEP
    
    # 3. Filter Duplicates
    new_movies_to_process = []
    for m in candidates:
        if m['id'] not in existing_ids:
            new_movies_to_process.append(m)
    
    count = len(new_movies_to_process)
    print(f"New additions found: {count}")

    if count == 0:
        return

    # 4. Compute & Ingest
    print(f"Processing {count} new movies...")
    
    for m in new_movies_to_process:
        # Resolve Genre IDs to Names
        # m['genre_ids'] might look like [28, 12]
        # We turn that into "Action Adventure"
        current_genres = [genre_map.get(gid, '') for gid in m.get('genre_ids', [])]
        genre_str = " ".join(current_genres)
        
        # Build the Soup with Tags
        # Structure: Title + Title + Genres + Overview
        soup = f"{m['title']} {m['title']} {genre_str} {m.get('overview', '')}"
        
        movie_data = {
            'id': m['id'],
            'title': m['title'],
            'soup': soup
        }
        
        rec.add_new_movie(movie_data)
        print(f" - Added: {m['title']} ({genre_str})")
        
    # 5. Save
    print("Saving updated index to disk...")
    rec.save('models/')
    print("Update Complete!")

if __name__ == "__main__":
    run_weekly_update()