from src.recommender import MovieRecommender

rec = MovieRecommender()
rec.load('models/')


# Assuming 'rec' is your loaded MovieRecommender instance

# Example 1: Vague description
print(rec.recommend_on_text("comedy about a group of friends going on an adventure"))


# # Example 2: Specific vibe
# print(rec.recommend_on_text("mind bending sci-fi about dreams within dreams"))
# # Expected: Inception

# # Example 3: Emotional query
# print(rec.recommend_on_text("sad movie that makes me cry about a dog"))
# # Expected: Hachi: A Dog's Tale or Marley & Me