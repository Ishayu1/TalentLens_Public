
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os
from streamlit.grok_utils import extract_skills_with_grok
# Note: we assume streamlit is in the same directory or grok_utils is accessible.
# Since grok_utils is currently in streamlit/, we might need to adjust or move it.

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
    input_mode: str = "Skills" # Added input_mode
    api_key: str | None = None # Added api_key

@app.post("/search")
async def search_resumes(request: SearchRequest):
    query = request.query
    skill_filters = []

    # --- Backend Grok Skill Extraction ---
    if request.input_mode == "Job Description":
        extracted_skills = extract_skills_with_grok(query, api_key=request.api_key)
        if extracted_skills:
            skill_filters = extracted_skills

    # Create query embedding
    query_embedding = model.encode([query])
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
