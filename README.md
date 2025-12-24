---
title: CineMatch API
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
---

# 🎬 CineMatch API

**CineMatch** is an intelligent, content-based movie recommendation engine powered by cutting-edge AI. It combines semantic search, vector embeddings, and personalization to deliver highly accurate movie recommendations tailored to user preferences.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Server](#running-the-server)
  - [API Endpoints](#api-endpoints)
  - [Examples](#examples)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Performance Considerations](#performance-considerations)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

### 1. **Semantic Search** 🔍
Search for movies using natural language queries. The system converts your text into a vector embedding and finds semantically similar movies.
- Example: *"A romantic movie about a sinking ship"* → Returns *Titanic*

### 2. **Vibe-Based Recommendations** 🎯
Search by combining tags (genres, themes) and descriptions for more refined results.
- Example: Tags: `["Sci-Fi", "Action"]`, Description: `"Robots fighting in space"` → Returns relevant matches

### 3. **Personalized Recommendations** 👤
Provide a list of movies you've liked, and the system averages their vectors to create a personalized profile, then recommends similar movies.
- Example: Liked: `["The Matrix", "Inception"]` → Get similar mind-bending films

### 4. **Content-Based Similarity** 🔗
Find movies similar to a specific title already in the database.
- Example: Similar to *"Inception"* → Returns *"Interstellar"*, *"The Matrix"*, etc.

### 5. **Rich Movie Metadata** 📊
Each movie includes:
- Director information
- Top 4 cast members
- Keywords (e.g., "time travel", "dystopia")
- Genres
- Plot overview
- IMDB ratings

### 6. **Incremental Learning** 📈
Add new movies to the system without retraining—updates are instant!

---

## 🏗️ Architecture

```
┌─────────────────┐
│   User Request  │
└────────┬────────┘
         │
    ┌────▼─────────────────────┐
    │   FastAPI Server         │
    │  (Endpoint Handler)       │
    └────┬──────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │  MovieRecommender Engine      │
    │  (FAISS + Vector Search)      │
    └────┬───────────────────────────┘
         │
    ┌────▼──────────────────────┐
    │  Embedding Model          │
    │  (SentenceTransformers)   │
    └────┬──────────────────────┘
         │
    ┌────▼──────────────────────┐
    │  FAISS Index              │
    │  (movie_index.faiss)      │
    └────┬──────────────────────┘
         │
    ┌────▼──────────────────────┐
    │  Movie Metadata           │
    │  (metadata.pkl)           │
    └──────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend Framework** | FastAPI | High-performance async API |
| **Vector Search** | FAISS | Fast similarity search on embeddings |
| **Embeddings** | SentenceTransformers (MiniLM-L6-v2) | Convert text to 384-dim vectors |
| **Data Source** | TMDB API | Movie metadata (titles, cast, genres, etc.) |
| **Data Processing** | Pandas, NumPy | Data cleaning & preprocessing |
| **Deployment** | Docker | Containerized deployment |
| **Python Version** | 3.9+ | Modern async/await support |

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- TMDB API Key (free, get it at [themoviedb.org](https://www.themoviedb.org/settings/api))
- ~2GB free disk space (for models and indices)

### Step 1: Clone & Setup

```bash
# Navigate to project directory
cd CineMatch

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

Create a `.env` file in the project root:

```
TMDB_API_KEY=your_api_key_here
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TMDB_API_KEY` | Your TMDB API key | `abc123xyz...` |

### Model Configuration

The default embedding model is **`all-MiniLM-L6-v2`** from SentenceTransformers:
- **Embedding Dimension**: 384
- **Speed**: Very fast (optimized for CPU)
- **Quality**: High for semantic similarity
- **Memory**: ~80MB

To use a different model, modify [recommender.py](src/recommender.py#L6) in the `MovieRecommender.__init__()` method.

---

## 🚀 Usage

### Running the Server

```bash
# Make sure your virtual environment is activated
python app.py
```

The server will start at `http://localhost:8000`

**API Documentation** (auto-generated Swagger UI):
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Data Ingestion

Before using the API, you need to populate the FAISS index with movies:

```bash
python src/ingest.py
```

This will:
1. Fetch ~50 high-quality movies from TMDB (popularity ≥ 7.0, votes ≥ 500)
2. Extract director, cast, and keywords for each movie
3. Generate embeddings
4. Save to `models/movie_index.faiss` and `models/metadata.pkl`

To reset and rebuild the index:

```python
# In src/ingest.py, modify the last line:
ingest_high_quality_movies(target_count=100, reset=True)  # reset=True to rebuild
```

---

## 📡 API Endpoints

### 1. **Health Check**
```
GET /
```

**Response:**
```json
{
  "status": "online and active!!!",
  "model_loaded": true
}
```

---

### 2. **Semantic Search** 🔍
```
POST /search
```

**Request:**
```json
{
  "query": "A romantic movie about a sinking ship",
  "k": 5
}
```

**Response:**
```json
[
  {
    "movie_id": 597,
    "title": "Titanic",
    "score": 0.856
  },
  {
    "movie_id": 285,
    "title": "The Poseidon Adventure",
    "score": 0.743
  }
]
```

---

### 3. **Vibe-Based Search** 🎯
```
POST /recommend/vibe
```

**Request:**
```json
{
  "tags": ["Sci-Fi", "Action", "Space"],
  "description": "Robots fighting in space with stunning visuals",
  "k": 10
}
```

**Response:**
```json
{
  "interpreted_query": "Sci-Fi Action Space Sci-Fi Action Space Robots fighting in space with stunning visuals",
  "results": [
    {
      "movie_id": 58,
      "title": "The Fifth Element",
      "score": 0.912
    }
  ]
}
```

---

### 4. **Personalized Recommendations** 👤
```
POST /recommend/user
```

**Request:**
```json
{
  "liked_movies": ["The Matrix", "Inception", "Interstellar"],
  "k": 5
}
```

**Response:**
```json
[
  {
    "movie_id": 27205,
    "title": "Oblivion",
    "score": 0.834
  },
  {
    "movie_id": 284054,
    "title": "Doctor Strange",
    "score": 0.798
  }
]
```

---

### 5. **Similar Movies** 🔗
```
GET /recommend/movie/{title}
```

**Example:**
```
GET /recommend/movie/Inception
```

**Response:**
```json
[
  {
    "movie_id": 38372,
    "title": "Interstellar",
    "score": 0.891
  },
  {
    "movie_id": 603,
    "title": "The Matrix",
    "score": 0.867
  }
]
```

---

### 6. **Admin: Trigger Background Update** 🔄
```
POST /admin/trigger-update
```

**Response:**
```json
{
  "message": "Update process started in background. Check server logs for progress."
}
```

This endpoint triggers background ingestion without blocking the API.

---

## 📝 Examples

### Example 1: Find Movies Similar to Your Favorite

```python
import requests

BASE_URL = "http://localhost:8000"

# Get movies similar to "The Matrix"
response = requests.get(f"{BASE_URL}/recommend/movie/The Matrix")
recommendations = response.json()

for movie in recommendations:
    print(f"{movie['title']} (Score: {movie['score']:.2f})")
```

### Example 2: Semantic Search with Natural Language

```python
response = requests.post(
    f"{BASE_URL}/search",
    json={
        "query": "A thrilling space adventure with amazing visuals",
        "k": 5
    }
)

for movie in response.json():
    print(f"✓ {movie['title']}")
```

### Example 3: Personalized Recommendations Based on History

```python
response = requests.post(
    f"{BASE_URL}/recommend/user",
    json={
        "liked_movies": ["Dune", "Blade Runner 2049", "Arrival"],
        "k": 10
    }
)

for movie in response.json():
    print(f"★ {movie['title']}")
```

---

## 📂 Project Structure

```
CineMatch/
├── app.py                    # Main FastAPI application
├── main.py                   # (Optional) Alternative entry point
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
├── .env                      # API keys (create this)
│
├── src/
│   ├── __init__.py
│   ├── recommender.py        # Core FAISS-based recommendation engine
│   ├── ingest.py             # TMDB data ingestion pipeline
│   └── preprocessing.py      # Data cleaning & feature engineering
│
├── models/
│   ├── movie_index.faiss     # FAISS index (generated after ingestion)
│   └── metadata.pkl          # Movie metadata dataframe (generated)
│
├── eda/
│   └── Untitled.ipynb        # Exploratory data analysis notebook
│
└── README.md                 # This file
```

---

## 🧠 How It Works

### The Embedding Pipeline

```
Raw Text Input (Movie Title + Metadata)
          ↓
    [SentenceTransformers]
          ↓
    384-Dimensional Vector
          ↓
    [L2 Normalization]
          ↓
    Normalized Vector (Unit Length)
          ↓
    [FAISS IndexFlatIP]
          ↓
    Stored in Index
```

### Recommendation Flow

1. **User provides query** (text, tags, or movie titles)
2. **Convert to vector** using SentenceTransformers
3. **Normalize vector** (for cosine similarity)
4. **FAISS search** finds K nearest neighbors in index
5. **Return results** with similarity scores

### Why This Approach?

- **Fast**: FAISS is optimized for billion-scale vector search
- **Accurate**: Semantic embeddings capture meaning, not just keywords
- **Scalable**: Can handle millions of movies
- **CPU-Friendly**: MiniLM model is tiny but effective
- **Incremental**: Add movies without retraining

---

## ⚡ Performance Considerations

### Indexing Speed
- **MiniLM Model**: ~100-200 movies/second on modern CPU
- **FAISS Indexing**: Instant for additions
- **Memory**: ~384 bytes per movie embedding

### Search Speed
- **Single Query**: 1-5ms
- **Batch Queries**: Linear time complexity O(n)
- **Max Practical Size**: 10+ million movies

### Optimization Tips

1. **Use Batch Processing**: Send multiple queries at once
2. **Tune k Parameter**: Lower k = faster results (typically k=5-10 is good)
3. **CPU**: The MiniLM model leverages BLAS libraries for speed
4. **GPU**: Optional—can speed up embedding generation 10x

---

## 🐳 Deployment

### Docker Build & Run

```bash
# Build image
docker build -t cinematch:latest .

# Run container
docker run -p 8000:8000 \
  -e TMDB_API_KEY=your_key \
  cinematch:latest
```

### Production Deployment

The project includes a `Dockerfile` configured for production use:
- **Base Image**: Python 3.9+
- **Port**: 8000 (configurable)
- **Entry**: `python app.py`

For production, consider:
- Using **Gunicorn** or **Uvicorn** with multiple workers
- Adding **Nginx** reverse proxy
- Implementing **authentication** (API keys)
- Using **cloud storage** for models (S3, GCS)

---

## 🐛 Troubleshooting

### Issue: "No model found" Error

**Solution**: Run data ingestion first:
```bash
python src/ingest.py
```

### Issue: TMDB API Key Invalid

**Solution**: Verify your `.env` file:
```bash
cat .env  # Check the key is there
```

### Issue: Out of Memory

**Solution**: Reduce batch size in [recommender.py](src/recommender.py#L18):
```python
batch_size = 32  # Lower from 64
```

### Issue: Slow Embedding Generation

**Solution**: 
- The MiniLM model is already optimized for CPU
- For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: CORS Errors

**Solution**: Already handled in [app.py](app.py#L15). The API allows all origins (`allow_origins=["*"]`). For production, restrict this:
```python
allow_origins=["https://yourdomain.com"]
```

---

## 📊 Dataset Information

**Movie Source**: The Movie Database (TMDB) API

**Filtering Criteria**:
- Minimum Rating: 7.0 / 10.0
- Minimum Vote Count: 500 votes
- Sorted by: Popularity (descending)

**Metadata Included**:
- Title
- Director
- Cast (top 4 actors)
- Keywords
- Genres
- Overview / Plot
- Vote Average

---

## 🔮 Future Enhancements

- [ ] User authentication & API key management
- [ ] Collaborative filtering (user-user similarity)
- [ ] Real-time model updates with webhooks
- [ ] Advanced filtering (year, rating, runtime)
- [ ] Movie rating & feedback loop for model improvement
- [ ] Multi-language support
- [ ] Mobile app integration

---

## 📄 License

This project is open source. Feel free to modify and extend it!

---

## 💬 Support

For issues, questions, or contributions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [API Documentation](http://localhost:8000/docs)
3. Examine the source code in `src/` directory

---

**Enjoy discovering your next favorite movie! 🍿🎬**