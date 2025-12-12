---
title: Cine Match Api
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
---

# CineMatch API

This is the backend for the CineMatch recommendation system.
It runs a FastAPI server using FAISS and SentenceTransformers.

## Endpoints
- POST `/search`: Semantic search
- POST `/recommend/vibe`: Search by tags + description
- POST `/recommend/user`: Personalized history-based recommendations