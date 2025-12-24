from src.recommender import MovieRecommender

def test_vibe():
    # 1. Load the new brain
    print("🧠 Loading the new high-quality brain...")
    rec = MovieRecommender()
    rec.load('models')
    # print(f"✅ Loaded {len(rec.df)} movies.\n")

    # 2. Define the Vibe
    description = "christpher nolan style space adventure with mind bending visuals"
    tags = ["Science Fiction", "Drama"]
    
    # COMBINE THEM: Since your function only takes text, we mix them together.
    # "Science Fiction Drama A space adventure..."
    full_query = f"{' '.join(tags)} {description}"

    print(f"🔎 Searching for: '{full_query}'")
    print("-" * 50)

    # 3. Get Recommendations (Using YOUR function name)
    results = rec.recommend_on_text(full_query, k=5)

    # 4. Print results
    for i, movie in enumerate(results):
        print(f"{i+1}. {movie['title']} (Score: {movie['score']:.2f})")

if __name__ == "__main__":
    test_vibe()