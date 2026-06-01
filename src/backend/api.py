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

app = FastAPI(
    title="TalentLens API",
    description="Backend API for DS3 TalentLens resume search.",
    version="1.0.0",
)

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

engine = None


def get_engine():
    global engine

    if engine is None:
        # Important for Cloud Run:
        # Do not require semantic FAISS/reranker startup validation.
        # This allows the API to boot even if Hugging Face/model cache is unavailable.
        engine = SearchEngine(strict_startup=False)

    return engine


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

    try:
        search_engine = get_engine()

        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            input_mode=request.input_mode,
            recruiter_company=request.recruiter_company,
            recruiter_job_title=request.recruiter_job_title,
        )

        return {
            "parsed_job_description": search_engine.last_query_analysis
            if request.input_mode == "Job Description"
            else None,
            "engine_status": {
                "demo_mode": getattr(search_engine, "demo_mode", None),
                "mode_label": getattr(search_engine, "mode_label", None),
                "retrieval_backend": getattr(search_engine, "retrieval_backend", None),
            },
            "results": [
                {
                    "rank": result.rank,
                    "filename": result.filename,
                    "candidate_id": result.candidate_id,
                    "score": safe_float(result.score),
                    "semantic_score": safe_float(result.semantic_score),
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
                    "grok_fit_score": safe_float(result.grok_fit_score),
                    "grok_resume_quality_score": safe_float(
                        result.grok_resume_quality_score
                    ),
                    "grok_summary": result.grok_summary,
                    "grok_matched_requirements": result.grok_matched_requirements,
                    "grok_missing_requirements": result.grok_missing_requirements,
                    "grok_weakness_flags": result.grok_weakness_flags,
                }
                for result in results
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"TalentLens search failed: {str(exc)}",
        )


@app.get("/health")
async def health():
    if engine is None:
        return {
            "status": "healthy",
            "message": "TalentLens API is running",
            "engine_loaded": False,
        }

    return {
        "status": "healthy",
        "message": "TalentLens API is running",
        "engine_loaded": True,
        "num_resumes": getattr(engine, "resume_count", None),
        "demo_mode": getattr(engine, "demo_mode", None),
        "mode_label": getattr(engine, "mode_label", None),
        "retrieval_backend": getattr(engine, "retrieval_backend", None),
    }