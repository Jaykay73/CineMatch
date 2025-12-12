from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn

# Import our custom modules
from src.recommender import MovieRecommender
from src.ingest import run_weekly_update

# --- Configuration ---
app = FastAPI(
    title="CineMatch API",
    description="A content-based movie recommender using FAISS & Transformers.",
    version="1.0.0"
)

# Enable CORS (Allows your future mobile app/website to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domain
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
        print(f" [INFO] Model loaded. Index contains {rec_engine.index.ntotal} movies.")
    else:
        print(" [WARNING] No model found at 'models/'. API will return errors until model is built.")

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
    movie_id: int
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
    return {"status": "online and active!!!", "model_loaded": rec_engine is not None}

@app.post("/search", response_model=List[MovieResponse])
def search_movies(request: SearchRequest):
    """
    Semantic Search: Convert query to vector -> Find nearest movies.
    """
    check_model()
    results = rec_engine.recommend_on_text(request.query, k=request.k)
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

    results = rec_engine.recommend_on_text(query_soup, k=request.k)
    
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
    
    # Handle case where no movies were matched
    if isinstance(results, str): 
        raise HTTPException(status_code=404, detail=results)
        
    return results

@app.get("/recommend/movie/{title}", response_model=List[MovieResponse])
def recommend_similar_movie(title: str):
    """
    Classic 'Users who liked X also liked Y' (Content-based version).
    """
    check_model()
    results = rec_engine.recommend_by_movie(title)
    
    if isinstance(results, str):
        raise HTTPException(status_code=404, detail=results)
        
    return results

# --- Admin / Operations ---

def background_update_task():
    """
    Runs the ingestion script and reloads the model in memory.
    """
    print(" [BACKGROUND] Starting update process...")
    # 1. Run the ingestion (Fetch from TMDB + Save to Disk)
    run_weekly_update()
    
    # 2. Reload the model in memory so the API sees the new movies immediately
    print(" [BACKGROUND] Reloading model into RAM...")
    rec_engine.load('models/')
    print(" [BACKGROUND] Update complete. Model reloaded.")

@app.post("/admin/trigger-update")
def trigger_update(background_tasks: BackgroundTasks):
    """
    Manually triggers the 'Weekly Update' logic. 
    Runs in the background so the API doesn't freeze.
    """
    background_tasks.add_task(background_update_task)
    return {"message": "Update process started in background. Check server logs for progress."}

# --- Entry Point ---
if __name__ == "__main__":
    # Use this for debugging. In production, use the command line.
    uvicorn.run(app, host="0.0.0.0", port=8000)