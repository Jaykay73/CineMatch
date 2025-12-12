import pandas as pd
import ast
import numpy as np

def clean_data(x):
    """
    Helper to convert stringified lists like "[{'name': 'Action'}]" 
    into a simple string "Action"
    """
    if isinstance(x, str):
        try:
            # Safely evaluate the string as a Python list/dict
            item_list = ast.literal_eval(x)
            if isinstance(item_list, list):
                # Extract the 'name' key from each dict in the list
                return ' '.join([i['name'] for i in item_list if 'name' in i])
        except (ValueError, SyntaxError):
            return ""
    return ""

def parse_features(data_path):
    print(f"Loading data from {data_path}...")
    
    # 1. Load Data
    # 'on_bad_lines' skips the few corrupted rows in this specific dataset
    df = pd.read_csv(data_path, low_memory=False)
    
    # 2. Filter Bad IDs
    # This dataset has a known bug where some IDs are dates (e.g., '1995-10-20')
    # We force 'id' to numeric and drop rows that fail
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['id'])
    df['id'] = df['id'].astype(int)

    # 3. Fill NaNs
    df['title'] = df['title'].fillna('')
    df['overview'] = df['overview'].fillna('')
    df['tagline'] = df['tagline'].fillna('')
    df['genres'] = df['genres'].fillna('[]')

    print("Parsing genres (this might take a moment)...")
    # 4. Clean Genres
    df['genre_names'] = df['genres'].apply(clean_data)

    # 5. Create the "Soup"
    # We combine Title (2x weight), Tagline, Overview, and Genres
    def create_soup(x):
        return f"{x['title']} {x['title']} {x['tagline']} {x['overview']} {x['genre_names']}"

    df['soup'] = df.apply(create_soup, axis=1)

    # 6. Return only what we need to save memory
    final_df = df[['id', 'title', 'soup']].reset_index(drop=True)
    
    print(f"Cleaned data: {len(final_df)} movies ready for embedding.")
    return final_df