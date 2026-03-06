"""
Search engine — loads the FAISS index + sentence-transformer model and
exposes a simple `search()` API consumed by the Streamlit UI.

If the pipeline artifacts (FAISS index, metadata JSON) haven't been generated
yet, the engine falls back to **demo mode** with synthetic data so the UI can
still be developed and previewed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
try:
    from grok_utils import extract_skills_with_grok
except ImportError:
    from streamlit.grok_utils import extract_skills_with_grok

import numpy as np
import pandas as pd

from config import (
    CONFIG_JSON_PATH,
    DEFAULT_TOP_K,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    MEMBERS_CSV,
    METADATA_PATH,
    MIN_SCORE_THRESHOLD,
    MODEL_NAME,
    SKILL_SUGGESTIONS,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ResumeResult:
    rank: int
    filename: str
    score: float
    file_path: str
    text_preview: str
    source: str = ""
    full_name: str = ""
    major: str = ""
    graduation_year: str = ""
    resume_link: str = ""
    linkedin: str = ""
    github: str = ""
    matched_skills: list = field(default_factory=list)
    explanation: str = "" # Grok AI explanation


# ---------------------------------------------------------------------------
# Keyword skill matcher (supplements semantic search)
# ---------------------------------------------------------------------------
_SKILL_PATTERNS: dict[str, re.Pattern] = {}


def _build_skill_patterns():
    if _SKILL_PATTERNS:
        return
    for skill in SKILL_SUGGESTIONS:
        escaped = re.escape(skill)
        _SKILL_PATTERNS[skill] = re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def extract_matched_skills(text: str, query_skills: list[str]) -> list[str]:
    """Return the subset of *query_skills* that appear in *text*."""
    _build_skill_patterns()
    matched = []
    for skill in query_skills:
        pat = _SKILL_PATTERNS.get(skill)
        if pat and pat.search(text):
            matched.append(skill)
        elif skill.lower() in text.lower():
            matched.append(skill)
    return matched


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------
class SearchEngine:
    """Wraps FAISS index + sentence-transformers for semantic resume search."""

    def __init__(self):
        self.model = None
        self.index = None
        self.metadata: list[dict] = []
        self.members_df: Optional[pd.DataFrame] = None
        self.demo_mode = False
        self._load()

    # ----- loading ----------------------------------------------------------
    def _load(self):
        try:
            self._load_production()
        except Exception as exc:
            print(f"[SearchEngine] Could not load production artifacts: {exc}")
            print("[SearchEngine] Falling back to demo mode.")
            self._load_demo()

    def _load_production(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")

        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

        if MEMBERS_CSV.exists():
            self.members_df = pd.read_csv(MEMBERS_CSV)

        self.demo_mode = False

    def _load_demo(self):
        """Generate synthetic data so the UI can be previewed."""
        self.demo_mode = True

        if MEMBERS_CSV.exists():
            self.members_df = pd.read_csv(MEMBERS_CSV)

        demo_names = [
            ("Alice Chen", "Computer Science (B.S.)", "2026"),
            ("Bob Patel", "Data Science (B.S.)", "2027"),
            ("Carol Kim", "Computer Engineering (B.S.)", "2026"),
            ("David Lopez", "Mathematics (B.S.)", "2028"),
            ("Emily Zhang", "Electrical Engineering (B.S.)", "2027"),
            ("Frank Johnson", "Computer Science (B.S.)", "2026"),
            ("Grace Lee", "Data Science (B.S.)", "2027"),
            ("Hector Rivera", "Statistics (B.S.)", "2028"),
            ("Ivy Wang", "Computer Science (B.S.)", "2026"),
            ("Jake Thompson", "Computer Engineering (B.S.)", "2027"),
        ]

        skills_pool = [
            "Python, Machine Learning, TensorFlow, SQL, Pandas",
            "Java, React, Node.js, AWS, Docker",
            "C++, Computer Vision, PyTorch, CUDA, OpenCV",
            "R, Statistics, Tableau, Power BI, Excel",
            "Python, NLP, Transformers, BERT, spaCy",
            "JavaScript, TypeScript, React, GraphQL, MongoDB",
            "Python, Deep Learning, Keras, scikit-learn, NumPy",
            "Spark, Hadoop, Scala, Kafka, Airflow",
            "Swift, Kotlin, Firebase, REST APIs, Git",
            "Go, Kubernetes, Terraform, CI/CD, Linux",
        ]

        self.metadata = []
        for i, ((name, major, grad_year), skills) in enumerate(
            zip(demo_names, skills_pool)
        ):
            self.metadata.append(
                {
                    "filename": f"{name.replace(' ', '_').lower()}_resume.pdf",
                    "file_path": f"data/ds3/member_resumes/{name.replace(' ', '_').lower()}_resume.pdf",
                    "text": (
                        f"{name}\n{major} — Class of {grad_year}\n\n"
                        f"Skills: {skills}\n\n"
                        "Experience: Software Engineering Intern at Tech Corp. "
                        "Developed machine learning pipelines for data analysis. "
                        "Built RESTful APIs and deployed models to production. "
                        "Published research on natural language processing."
                    ),
                    "full_name": name,
                    "major": major,
                    "graduation_year": grad_year,
                }
            )

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_SCORE_THRESHOLD,
        skill_filters: list[str] | None = None,
        grad_year_filter: str | None = None,
        major_filter: str | None = None,
        input_mode: str = "Skills",
        api_key: str | None = None,
    ) -> list[ResumeResult]:
        if self.demo_mode:
            return self._search_demo(
                query, top_k, skill_filters, grad_year_filter, major_filter
            )
        return self._search_production(
            query, top_k, min_score, skill_filters, grad_year_filter, major_filter, input_mode, api_key
        )

    def _search_production(
        self,
        query: str,
        top_k: int,
        min_score: float,
        skill_filters: list[str] | None,
        grad_year_filter: str | None,
        major_filter: str | None,
        input_mode: str = "Skills",
        api_key: str | None = None,
    ) -> list[ResumeResult]:
        import faiss as _faiss

        # --- Backend Grok Skill Extraction ---
        if input_mode == "Job Description":
            extracted_skills = extract_skills_with_grok(query, api_key=api_key)
            if extracted_skills:
                # Merge with any existing filters
                if skill_filters:
                    skill_filters = list(set(skill_filters + extracted_skills))
                else:
                    skill_filters = extracted_skills

        query_embedding = self.model.encode([query]).astype("float32")
        _faiss.normalize_L2(query_embedding)

        # Increase fetch window to allow boosted resumes to climb from further down
        fetch_k = max(100, min(top_k * 10, len(self.metadata)))
        scores, indices = self.index.search(query_embedding, fetch_k)

        query_skills = [s.strip() for s in query.split(",") if s.strip()]
        if skill_filters:
            query_skills = list(set(query_skills + skill_filters))

        # Check for prominent companies in the query/JD to apply extra emphasis
        target_companies = ["amazon", "google", "meta", "microsoft", "apple", "netflix"]
        jd_companies = [c for c in target_companies if c in query.lower()]

        results: list[ResumeResult] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or score < min_score:
                continue
            meta = self.metadata[idx]
            text = meta.get("text", "")
            matched = extract_matched_skills(text, query_skills) if query_skills else []

            # --- Score Boosting ---
            final_score = float(score)
            if query_skills and matched:
                match_ratio = len(matched) / len(query_skills)
                final_score += (0.2 * match_ratio)
                
                # HUGE boost for target company match if it's the subject of the search
                for company in jd_companies:
                    # Check for variations like AWS for Amazon
                    is_match = company in text.lower() or (company == "amazon" and "aws" in text.lower())
                    
                    if is_match:
                        # massive boost to prioritize past experience at the same company
                        final_score += 0.3
                        # Extra boost for specific career-starting roles (Intern/SDE/Internship)
                        # We use multiline regex with a slightly larger window
                        role_pattern = rf"(intern|sde|engineer|researcher|analyst)[\s\S]{{0,100}}\b({company}|aws)\b"
                        if re.search(role_pattern, text.lower(), re.I):
                             final_score += 0.3
                             # Specific higher priority for Interns/SDEs per user request
                             if re.search(rf"(intern|sde)[\s\S]{{0,50}}\b({company}|aws)\b", text.lower(), re.I):
                                 final_score += 0.2
                        elif re.search(rf"\b({company}|aws)\b[\s\S]{{0,100}}\b(intern|sde|engineer|researcher|analyst)\b", text.lower(), re.I):
                             final_score += 0.3
                             # Specific higher priority for Interns/SDEs per user request
                             if re.search(rf"\b({company}|aws)\b[\s\S]{{0,50}}\b(intern|sde)\b", text.lower(), re.I):
                                 final_score += 0.2
                
                final_score = min(final_score, 1.8) # Allow ranking to go higher
                         

            member_info = self._lookup_member(meta.get("filename", ""), text)

            if grad_year_filter and member_info.get("graduation_year", "") != grad_year_filter:
                continue
            if major_filter and major_filter.lower() not in member_info.get("major", "").lower():
                continue

            results.append(
                ResumeResult(
                    rank=0,
                    filename=meta.get("filename", ""),
                    score=final_score,
                    file_path=meta.get("file_path", ""),
                    text_preview=text[:400],
                    source=meta.get("source", "ds3_members"),
                    full_name=member_info.get("full_name", meta.get("filename", "")),
                    major=member_info.get("major", ""),
                    graduation_year=member_info.get("graduation_year", ""),
                    resume_link=member_info.get("resume_link", ""),
                    linkedin=member_info.get("linkedin", ""),
                    github=member_info.get("github", ""),
                    matched_skills=matched,
                )
            )

        # Re-sort results based on the boosted score and take top_k
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]

        for i, r in enumerate(results, 1):
            r.rank = i

        return results

    def _search_demo(
        self,
        query: str,
        top_k: int,
        skill_filters: list[str] | None,
        grad_year_filter: str | None,
        major_filter: str | None,
    ) -> list[ResumeResult]:
        """Simple text-overlap scoring for demo purposes."""
        query_lower = query.lower()
        query_tokens = set(re.findall(r"\w+", query_lower))

        query_skills = [s.strip() for s in query.split(",") if s.strip()]
        if skill_filters:
            query_skills = list(set(query_skills + skill_filters))
        if not query_tokens and skill_filters:
            for sf in skill_filters:
                query_tokens.update(re.findall(r"\w+", sf.lower()))

        scored = []
        for meta in self.metadata:
            text = meta.get("text", "")
            text_lower = text.lower()
            text_tokens = set(re.findall(r"\w+", text_lower))
            overlap = len(query_tokens & text_tokens)
            score = overlap / max(len(query_tokens), 1)

            grad_year = meta.get("graduation_year", "")
            major = meta.get("major", "")

            if grad_year_filter and str(grad_year) != grad_year_filter:
                continue
            if major_filter and major_filter.lower() not in major.lower():
                continue

            matched = extract_matched_skills(text, query_skills) if query_skills else []
            if skill_filters and not matched:
                score *= 0.3

            scored.append((meta, score, matched))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[ResumeResult] = []
        for i, (meta, score, matched) in enumerate(scored[:top_k], 1):
            results.append(
                ResumeResult(
                    rank=i,
                    filename=meta.get("filename", ""),
                    score=min(score, 1.0),
                    file_path=meta.get("file_path", ""),
                    text_preview=meta.get("text", "")[:400],
                    source="demo",
                    full_name=meta.get("full_name", meta.get("filename", "")),
                    major=meta.get("major", ""),
                    graduation_year=meta.get("graduation_year", ""),
                    resume_link="",
                    linkedin="",
                    github="",
                    matched_skills=matched,
                )
            )
        return results

    # ----- helpers ----------------------------------------------------------
    def _lookup_member(self, filename: str, text: str = "") -> dict:
        """Try to match a resume filename or content to a row in members.csv."""
        if self.members_df is None:
            return {}
        
        # 1. Try filename stem match
        name_stem = Path(filename).stem.replace("_", " ").replace("-", " ").lower()
        for _, row in self.members_df.iterrows():
            full_name = str(row.get("Full Name", "")).lower()
            if full_name and full_name in name_stem:
                return self._row_to_meta(row)
        
        # 2. Try matching name within the first part of the text (heuristic for resumes)
        if text:
            text_head = text[:500].lower()
            for _, row in self.members_df.iterrows():
                full_name = str(row.get("Full Name", "")).lower()
                if full_name and full_name in text_head:
                    return self._row_to_meta(row)
                    
        return {}

    def _row_to_meta(self, row) -> dict:
        return {
            "full_name": row.get("Full Name", ""),
            "major": row.get("Major", ""),
            "graduation_year": str(row.get("Graduation Year", "")),
            "resume_link": row.get("Resume Link", ""),
            "linkedin": row.get("Linkedin Link", ""),
            "github": row.get("Github Link", ""),
        }

    @property
    def resume_count(self) -> int:
        return len(self.metadata)

    def get_unique_majors(self) -> list[str]:
        if self.members_df is not None:
            return sorted(self.members_df["Major"].dropna().unique().tolist())
        return sorted({m.get("major", "") for m in self.metadata if m.get("major")})

    def get_unique_grad_years(self) -> list[str]:
        if self.members_df is not None:
            return sorted(
                self.members_df["Graduation Year"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        return sorted({str(m.get("graduation_year", "")) for m in self.metadata if m.get("graduation_year")})
