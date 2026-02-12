
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

app = FastAPI()

# Load model and index on startup
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('resume_index.faiss')
with open('member_resumes_metadata.json', 'r') as f:
    resumes_metadata = json.load(f)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    min_score: float = 0.0

@app.post("/search")
async def search_resumes(request: SearchRequest):
    # Create query embedding
    query_embedding = model.encode([request.query])
    query_embedding = query_embedding.astype('float32')
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(query_embedding, request.top_k)

    # Format results
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if score >= request.min_score:
            resume = resumes_metadata[idx]
            results.append({
                'filename': resume['filename'],
                'score': float(score),
                'file_path': resume['file_path']
            })

    return {'results': results}

@app.get("/health")
async def health():
    return {"status": "healthy", "num_resumes": len(resumes_metadata)}

# Run with: uvicorn api:app --reload
