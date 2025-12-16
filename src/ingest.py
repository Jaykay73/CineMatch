import requests
import pandas as pd
import time
import os
from src.recommender import MovieRecommender

# CONFIGURATION
API_KEY =  os.getenv('API_KEY')
BASE_URL = "https://api.themoviedb.org/3"

def get_genre_map():
    """Fetches Genre IDs -> Names (e.g., 28 -> 'Action')"""
    url = f"{BASE_URL}/genre/movie/list"
    params = {'api_key': API_KEY, 'language': 'en-US'}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        return {g['id']: g['name'] for g in data.get('genres', [])}
    except Exception as e:
        print(f"Failed to load genres: {e}")
        return {}

def bulk_ingest(target_count=2000):
    print(f"--- Starting Bulk Ingestion of {target_count} Movies ---")
    
    # 1. Load Existing Brain
    rec = MovieRecommender()
    try:
        rec.load('models/')
        existing_ids = set(rec.movies['id'].values)
        print(f"Loaded existing index with {len(existing_ids)} movies.")
    except:
        print("No existing index found. Starting fresh.")
        existing_ids = set()

    # 2. Setup
    genre_map = get_genre_map()
    movies_added = 0
    page = 1
    
    # TMDB 'Top Rated' or 'Popular' are best for building a good catalog
    # Switch endpoint to 'top_rated' to get quality movies, or keep 'popular'
    endpoint = "popular" 

    while movies_added < target_count:
        print(f"Fetching page {page}...")
        url = f"{BASE_URL}/movie/{endpoint}"
        params = {
            'api_key': API_KEY,
            'language': 'en-US',
            'page': page
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"Error fetching page {page}: {response.status_code}")
                break
                
            results = response.json().get('results', [])
            
            # If we run out of pages
            if not results:
                print("No more movies available from API.")
                break

            # Process this batch of 20
            batch_added = 0
            for m in results:
                # Stop if we hit the limit mid-page
                if movies_added >= target_count:
                    break

                # Skip duplicates
                if m['id'] in existing_ids:
                    continue

                # Build Soup
                current_genres = [genre_map.get(gid, '') for gid in m.get('genre_ids', [])]
                genre_str = " ".join(current_genres)
                soup = f"{m['title']} {m['title']} {genre_str} {m.get('overview', '')}"

                movie_data = {
                    'id': m['id'],
                    'title': m['title'],
                    'soup': soup
                }

                # Add to Model
                rec.add_new_movie(movie_data)
                existing_ids.add(m['id'])
                
                movies_added += 1
                batch_added += 1

            print(f" - Page {page}: Added {batch_added} new movies. (Total: {len(existing_ids)})")

            # SAVE CHECKPOINT every 5 pages (approx 100 movies)
            if page % 5 == 0 and batch_added > 0:
                print(" >> Saving checkpoint to disk...")
                rec.save('models/')

            page += 1
            
            # Be nice to the API
            time.sleep(0.2)

        except Exception as e:
            print(f"Critical Error: {e}")
            break

    # Final Save
    print("--- Ingestion Complete ---")
    print(f"Total movies in database: {len(existing_ids)}")
    rec.save('models/')

if __name__ == "__main__":
    # You can change this number to whatever you want
    bulk_ingest(target_count=50)