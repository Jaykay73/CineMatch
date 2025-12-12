from src.preprocessing import parse_features
from src.recommender import MovieRecommender
import time

# 1. Setup paths
# Make sure your csv is exactly at this path
csv_path = 'data/movies_metadata.csv' 

# 2. Run Preprocessing
# This handles the 45k rows and fixes the bad ID bug
df = parse_features(csv_path)

# 3. Initialize Recommender
# Using the CPU-optimized class we discussed
rec = MovieRecommender()

# 4. Generate Embeddings
# This is the heavy step. Go grab a coffee.
start_time = time.time()
rec.preprocess_and_embed(df)
print(f"Total time taken: {(time.time() - start_time)/60:.2f} minutes")

# 5. Save the result so you don't have to wait next time
rec.save('models/')

# 6. Quick Test
print("\n--- Test Recommendation ---")
print(rec.recommend_by_movie("Jumanji"))