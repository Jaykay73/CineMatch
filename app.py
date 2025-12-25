from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import uvicorn

# Import our custom modules
from src.recommender import MovieRecommender
from src.ingest import ingest_high_quality_movies

# --- Configuration ---
app = FastAPI(
    title="CineMatch API",
    description="A content-based movie recommender using FAISS & Transformers.",
    version="2.0.0"
)

# Enable CORS (Allows your frontend to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aether-match.vercel.app"],  # In production, replace with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State for the AI Model
rec_engine: Optional[MovieRecommender] = None

# --- Lifespan Events (Startup/Shutdown) ---
@app.on_event("startup")
async def startup_event():
    """
    Load the heavy AI model once when the server starts.
    """
    global rec_engine
    rec_engine = MovieRecommender()
    
    if os.path.exists('models/movie_index.faiss'):
        print(" [INFO] Loading AI Model from disk...")
        rec_engine.load('models/')
        # Safety check for index existence
        count = rec_engine.index.ntotal if rec_engine.index else 0
        print(f" [INFO] Model loaded. Index contains {count} movies.")
    else:
        print(" [WARNING] No model found at 'models/'. API will return errors until ingestion is run.")

# --- Pydantic Data Models (Schema) ---
class SearchRequest(BaseModel):
    query: str
    k: int = 10

class VibeRequest(BaseModel):
    tags: List[str] = []     # e.g., ["Sci-Fi", "90s"]
    description: str = ""    # e.g., "Robots fighting in space"
    k: int = 10

class UserHistoryRequest(BaseModel):
    liked_movies: List[str]  # e.g., ["The Matrix", "Inception"]
    k: int = 10

class MovieResponse(BaseModel):
    # Updated to match the new Recommender output keys
    id: int
    title: str
    score: float

# --- Helper Function ---
def check_model():
    if not rec_engine or not rec_engine.index:
        raise HTTPException(status_code=503, detail="AI Model is not loaded. Run ingestion first.")

# --- API Endpoints ---

@app.get("/")
def health_check():
    """Simple check to see if server is running."""
    loaded = rec_engine is not None and rec_engine.index is not None
    return {"status": "online", "model_loaded": loaded}

@app.post("/search", response_model=List[MovieResponse])
def search_movies(request: SearchRequest):
    """
    Semantic Search: Convert query to vector -> Find nearest movies.
    Now includes Guardrails automatically via the Recommender class.
    """
    check_model()
    results = rec_engine.recommend(request.query, k=request.k)
    return results

@app.post("/recommend/vibe", response_model=dict)
def vibe_check(request: VibeRequest):
    """
    Recommends based on a mix of Tags and Description.
    """
    check_model()
    
    # Construct the "Soup"
    # We repeat tags to give them more weight in the vector space
    tag_str = " ".join(request.tags) * 2
    query_soup = f"{tag_str} {request.description}".strip()
    
    if not query_soup:
        raise HTTPException(status_code=400, detail="Please provide at least one tag or description.")

    # We use the standard recommend method which now includes guardrails
    results = rec_engine.recommend(query_soup, k=request.k)
    
    return {
        "interpreted_query": query_soup,
        "results": results
    }

@app.post("/recommend/user", response_model=List[MovieResponse])
def recommend_for_user(request: UserHistoryRequest):
    """
    Takes a list of movie titles the user likes, averages their vectors, 
    and finds similar movies.
    """
    check_model()
    results = rec_engine.recommend_for_user(request.liked_movies, k=request.k)
    
    # Handle empty results (e.g., none of the liked movies were in our DB)
    if not results: 
        return []
        
    return results

@app.get("/recommend/movie/{title}", response_model=List[MovieResponse])
def recommend_similar_movie(title: str):
    """
    Finds movies similar to a specific title.
    We reuse 'recommend_for_user' logic passing a single movie.
    """
    check_model()
    # Treat a single movie as a "User History" of 1
    results = rec_engine.recommend_for_user([title], k=10)
    
    if not results:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found in database.")
        
    return results

# --- Admin / Operations ---

def background_update_task():
    """
    Runs the ingestion script and reloads the model in memory.
    """
    print(" [BACKGROUND] Starting update process...")
    
    # 1. Run Ingest: Append 50 new movies, DO NOT RESET
    try:
        ingest_high_quality_movies(target_count=50, reset=False)
        print(" [BACKGROUND] Ingestion complete.")
    except Exception as e:
        print(f" [ERROR] Ingestion failed: {e}")
        return

    # 2. Reload the model in memory so the API sees the new movies immediately
    print(" [BACKGROUND] Reloading model into RAM...")
    rec_engine.load('models/')
    print(" [BACKGROUND] Update complete. Model reloaded.")

@app.post("/update") # Simplified endpoint name
def trigger_update(background_tasks: BackgroundTasks):
    """
    Manually triggers the 'Weekly Update' logic. 
    Runs in the background so the API doesn't freeze.
    """
    background_tasks.add_task(background_update_task)
    return {"message": "Update process started in background (Append Mode)."}

# --- Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
