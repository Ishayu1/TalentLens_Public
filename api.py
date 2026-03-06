from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys

STREAMLIT_DIR = Path(__file__).resolve().parent / "streamlit"
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))

from search import SearchEngine

app = FastAPI()

# Load shared search engine on startup so API and Streamlit rank identically.
engine = SearchEngine()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    min_score: float = 0.0
    input_mode: str = "Skills"
    api_key: str | None = None

@app.post("/search")
async def search_resumes(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    results = engine.search(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        input_mode=request.input_mode,
        api_key=request.api_key,
    )

    return {
        "results": [
            {
                "rank": result.rank,
                "filename": result.filename,
                "score": float(result.score),
                "semantic_score": float(result.semantic_score),
                "recruiter_score": float(result.recruiter_score),
                "resume_quality_score": float(result.resume_quality_score),
                "file_path": result.file_path,
                "full_name": result.full_name,
                "major": result.major,
                "graduation_year": result.graduation_year,
                "matched_skills": result.matched_skills,
                "explanation": result.explanation,
                "recruiter_breakdown": result.recruiter_breakdown,
                "resume_quality_breakdown": result.resume_quality_breakdown,
                "resume_flags": result.resume_flags,
                "hard_fail_flags": result.hard_fail_flags,
            }
            for result in results
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "num_resumes": engine.resume_count,
        "demo_mode": engine.demo_mode,
        "mode_label": engine.mode_label,
    }

# Run with: uvicorn api:app --reload
