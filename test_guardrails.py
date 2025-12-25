from src.recommender import MovieRecommender

rec = MovieRecommender()
rec.load('models')

print("--- TESTING HAPPY MODE ---")
# This text triggers the "Comedy" guardrail which bans "Crime/Thriller"
results = rec.recommend("horror movie", k=15)

for m in results:
    print(f"✅ {m['title']}")