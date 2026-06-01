from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UI_DIR = PROJECT_ROOT / "src" / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from search import SearchEngine

app = FastAPI()
engine = SearchEngine(strict_startup=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://ds3atucsd.com",
        "https://www.ds3atucsd.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "TalentLens API is running",
        "docs": "/docs",
        "health": "/health",
        "search": "/search",
    }


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    min_score: float = 0.0
    input_mode: str = "Skills"
    recruiter_company: str | None = None
    recruiter_job_title: str | None = None


@app.post("/search")
async def search_resumes(request: SearchRequest):
    if request.input_mode not in {"Skills", "Job Description"}:
        raise HTTPException(
            status_code=400,
            detail="input_mode must be either 'Skills' or 'Job Description'.",
        )
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    results = engine.search(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        input_mode=request.input_mode,
        recruiter_company=request.recruiter_company,
        recruiter_job_title=request.recruiter_job_title,
    )

    return {
        "parsed_job_description": engine.last_query_analysis
        if request.input_mode == "Job Description"
        else None,
        "results": [
            {
                "rank": result.rank,
                "filename": result.filename,
                "candidate_id": result.candidate_id,
                "score": float(result.score),
                "semantic_score": float(result.semantic_score),
                "file_path": result.file_path,
                "full_name": result.full_name,
                "major": result.major,
                "graduation_year": result.graduation_year,
                "matched_skills": result.matched_skills,
                "top_evidence_chunks": result.top_evidence_chunks,
                "hard_filter_status": result.hard_filter_status,
                "ranking_details": result.ranking_details,
                "page_count": result.page_count,
                "company_match_status": result.company_match_status,
                "grok_status": result.grok_status,
                "grok_fit_score": float(result.grok_fit_score),
                "grok_resume_quality_score": float(result.grok_resume_quality_score),
                "grok_summary": result.grok_summary,
                "grok_matched_requirements": result.grok_matched_requirements,
                "grok_missing_requirements": result.grok_missing_requirements,
                "grok_weakness_flags": result.grok_weakness_flags,
            }
            for result in results
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "num_resumes": engine.resume_count,
        "demo_mode": engine.demo_mode,
        "mode_label": engine.mode_label,
        "retrieval_backend": engine.retrieval_backend,
    }
