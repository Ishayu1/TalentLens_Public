from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
import traceback

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


def get_resume_metadata(search_engine, filename: str | None) -> dict:
    """
    Look up the original resume metadata for fields that may not be carried
    directly on the ResumeResult object, such as resume_link, linkedin, github.
    """
    if not filename:
        return {}

    metadata_by_filename = getattr(search_engine, "resume_metadata_by_filename", {}) or {}

    if filename in metadata_by_filename:
        return metadata_by_filename.get(filename) or {}

    for item in getattr(search_engine, "resume_metadata", []) or []:
        if item.get("filename") == filename:
            return item

    return {}


def serialize_result(search_engine, result):
    """
    Convert an internal ResumeResult object into a frontend-safe API payload.
    Includes metadata links from resume_metadata when they are not present
    directly on the result object.
    """
    metadata = get_resume_metadata(search_engine, getattr(result, "filename", None))

    return {
        "rank": getattr(result, "rank", None),
        "filename": getattr(result, "filename", None),
        "candidate_id": getattr(result, "candidate_id", None),
        "score": safe_float(getattr(result, "score", None)),
        "semantic_score": safe_float(getattr(result, "semantic_score", None)),
        "file_path": getattr(result, "file_path", None),
        "full_name": getattr(result, "full_name", None) or metadata.get("full_name"),
        "major": getattr(result, "major", None) or metadata.get("major"),
        "graduation_year": getattr(result, "graduation_year", None)
        or metadata.get("graduation_year"),

        # Important recruiter-facing links
        "resume_link": getattr(result, "resume_link", None) or metadata.get("resume_link"),
        "linkedin": getattr(result, "linkedin", None) or metadata.get("linkedin"),
        "github": getattr(result, "github", None) or metadata.get("github"),

        "matched_skills": getattr(result, "matched_skills", []) or [],
        "top_evidence_chunks": getattr(result, "top_evidence_chunks", []) or [],
        "hard_filter_status": getattr(result, "hard_filter_status", None),
        "ranking_details": getattr(result, "ranking_details", None),
        "page_count": getattr(result, "page_count", None),
        "company_match_status": getattr(result, "company_match_status", None),
        "grok_status": getattr(result, "grok_status", None),
        "grok_fit_score": safe_float(getattr(result, "grok_fit_score", None)),
        "grok_resume_quality_score": safe_float(
            getattr(result, "grok_resume_quality_score", None)
        ),
        "grok_summary": getattr(result, "grok_summary", None),
        "grok_matched_requirements": getattr(result, "grok_matched_requirements", [])
        or [],
        "grok_missing_requirements": getattr(result, "grok_missing_requirements", [])
        or [],
        "grok_weakness_flags": getattr(result, "grok_weakness_flags", []) or [],
    }


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

        raw_results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            input_mode=request.input_mode,
            recruiter_company=request.recruiter_company,
            recruiter_job_title=request.recruiter_job_title,
        )

        results_payload = [
            serialize_result(search_engine, result) for result in raw_results
        ]

        return {
            "parsed_job_description": search_engine.last_query_analysis
            if request.input_mode == "Job Description"
            else None,
            "engine_status": {
                "demo_mode": getattr(search_engine, "demo_mode", None),
                "mode_label": getattr(search_engine, "mode_label", None),
                "retrieval_backend": getattr(search_engine, "retrieval_backend", None),
            },
            "results": results_payload,
        }

    except Exception as exc:
        print("TalentLens search failed:", flush=True)
        print(traceback.format_exc(), flush=True)

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