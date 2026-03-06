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
from typing import Optional
import os
try:
    from grok_utils import (
        extract_skills_with_grok,
        assess_candidate_with_grok,
        assess_resume_quality_with_grok,
    )
except ImportError:
    from streamlit.grok_utils import (
        extract_skills_with_grok,
        assess_candidate_with_grok,
        assess_resume_quality_with_grok,
    )

import numpy as np
import pandas as pd

try:
    from config import (
        CONFIG_JSON_PATH,
        DATA_DIR,
        DEFAULT_TOP_K,
        EMBEDDING_DIM,
        FAISS_INDEX_PATH,
        MEMBER_RESUMES_DIR,
        MEMBERS_CSV,
        METADATA_PATH,
        MIN_SCORE_THRESHOLD,
        MODEL_NAME,
        PROJECT_ROOT,
        SKILL_SUGGESTIONS,
    )
except ImportError:
    from streamlit.config import (
        CONFIG_JSON_PATH,
        DATA_DIR,
        DEFAULT_TOP_K,
        EMBEDDING_DIM,
        FAISS_INDEX_PATH,
        MEMBER_RESUMES_DIR,
        MEMBERS_CSV,
        METADATA_PATH,
        MIN_SCORE_THRESHOLD,
        MODEL_NAME,
        PROJECT_ROOT,
        SKILL_SUGGESTIONS,
    )

RECRUITER_RERANK_LIMIT = 8
RECRUITER_RERANK_BASE_WEIGHT = 0.56
RECRUITER_RERANK_MATCH_WEIGHT = 0.24
RECRUITER_RERANK_QUALITY_WEIGHT = 0.20
RECRUITER_MAX_SCORE = 1.8
FALLBACK_RESUME_PATH = DATA_DIR / "processed" / "resumes_extracted.json"
FALLBACK_EMBEDDING_METADATA_PATH = DATA_DIR / "processed" / "resumes_with_embeddings.json"
FALLBACK_EMBEDDINGS_PATH = DATA_DIR / "processed" / "embeddings.npy"
LOCAL_MEMBER_RESUME_DIRS = [
    MEMBER_RESUMES_DIR,
    PROJECT_ROOT / "test" / "members",
]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "if",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "will", "with", "you", "your",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ResumeResult:
    rank: int
    filename: str
    score: float
    semantic_score: float
    file_path: str
    local_resume_path: str
    text_preview: str
    full_text: str = ""
    source: str = ""
    full_name: str = ""
    major: str = ""
    graduation_year: str = ""
    resume_link: str = ""
    linkedin: str = ""
    github: str = ""
    matched_skills: list = field(default_factory=list)
    explanation: str = ""  # Grok AI explanation
    recruiter_score: float = 0.0
    recruiter_breakdown: dict = field(default_factory=dict)
    resume_quality_score: float = 0.0
    resume_quality_breakdown: dict = field(default_factory=dict)
    resume_flags: list[str] = field(default_factory=list)
    hard_fail_flags: list[str] = field(default_factory=list)


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


def _tokenize_text(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9.+#-]*", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------
class SearchEngine:
    """Wraps FAISS index + sentence-transformers for semantic resume search."""

    def __init__(self):
        self.model = None
        self.index = None
        self.fallback_embeddings: np.ndarray | None = None
        self.fallback_term_frequencies: list[dict[str, int]] = []
        self.fallback_doc_lengths: list[int] = []
        self.fallback_idf: dict[str, float] = {}
        self.fallback_avg_doc_length: float = 0.0
        self.metadata: list[dict] = []
        self.members_df: Optional[pd.DataFrame] = None
        self.demo_mode = False
        self.mode_label = "Live"
        self.mode_banner = ""
        self._load()

    # ----- loading ----------------------------------------------------------
    def _load(self):
        try:
            self._load_production()
        except Exception as exc:
            print(f"[SearchEngine] Could not load production artifacts: {exc}")
            try:
                print("[SearchEngine] Falling back to DS3 resume text search.")
                self._load_ds3_fallback()
            except Exception as fallback_exc:
                print(f"[SearchEngine] Could not load DS3 fallback data: {fallback_exc}")
                print("[SearchEngine] Falling back to synthetic demo mode.")
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
        self.mode_label = "Live"
        self.mode_banner = ""

    def _load_ds3_fallback(self):
        if not FALLBACK_RESUME_PATH.exists():
            raise FileNotFoundError(f"Fallback resume data not found at {FALLBACK_RESUME_PATH}")

        with open(FALLBACK_RESUME_PATH, "r") as f:
            raw_resumes = json.load(f)

        if MEMBERS_CSV.exists():
            self.members_df = pd.read_csv(MEMBERS_CSV)

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(MODEL_NAME)
        except Exception as exc:
            print(f"[SearchEngine] Could not load fallback embedding model: {exc}")
            self.model = None

        embedding_map = self._load_fallback_embedding_map()
        ds3_resumes = []
        fallback_vectors = []
        for item in raw_resumes:
            if item.get("source") != "ds3_members":
                continue

            member_meta = item.get("metadata", {})
            filename = item.get("filename", "")
            ds3_resumes.append(
                {
                    "filename": filename,
                    "file_path": item.get("file_path", ""),
                    "text": item.get("text", ""),
                    "source": item.get("source", "ds3_members"),
                    "full_name": member_meta.get("full_name", ""),
                    "major": member_meta.get("major", ""),
                    "graduation_year": str(member_meta.get("graduation_year", "")),
                    "resume_link": member_meta.get("resume_link", ""),
                    "linkedin": member_meta.get("linkedin", ""),
                    "github": member_meta.get("github", ""),
                }
            )
            if embedding_map is not None:
                embedding = embedding_map.get(filename)
                if embedding is not None:
                    fallback_vectors.append(embedding)
                else:
                    fallback_vectors.append(np.zeros(EMBEDDING_DIM, dtype="float32"))

        if not ds3_resumes:
            raise ValueError("No DS3 resumes found in fallback resume data.")

        self.metadata = ds3_resumes
        if fallback_vectors and len(fallback_vectors) == len(self.metadata):
            self.fallback_embeddings = np.vstack(fallback_vectors).astype("float32")
            norms = np.linalg.norm(self.fallback_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.fallback_embeddings = self.fallback_embeddings / norms
        else:
            self.fallback_embeddings = None
        self._prepare_fallback_text_index()
        self.demo_mode = True
        self.mode_label = "Fallback"
        self.mode_banner = (
            "FAISS artifacts are unavailable, so TalentLens is using real DS3 resumes "
            "with fallback search."
        )

    def _load_demo(self):
        """Generate synthetic data so the UI can be previewed."""
        self.demo_mode = True
        self.mode_label = "Demo"
        self.mode_banner = (
            "Pipeline artifacts and fallback DS3 data are unavailable. "
            "Showing synthetic resumes for UI preview only."
        )

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
            if self.mode_label == "Fallback":
                return self._search_fallback(
                    query,
                    top_k,
                    min_score,
                    skill_filters,
                    grad_year_filter,
                    major_filter,
                    input_mode,
                    api_key,
                )
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
                    semantic_score=float(score),
                    file_path=meta.get("file_path", ""),
                    local_resume_path=self._resolve_resume_path(
                        meta.get("filename", ""),
                        meta.get("file_path", ""),
                    ),
                    text_preview=text[:400],
                    full_text=text,
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

        results.sort(key=lambda x: x.score, reverse=True)
        if input_mode == "Job Description":
            results = self._apply_recruiter_reranking(results, query, api_key)
        results = results[:top_k]

        for i, r in enumerate(results, 1):
            r.rank = i

        return results

    def _search_fallback(
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
        if input_mode == "Job Description":
            extracted_skills = extract_skills_with_grok(query, api_key=api_key)
            if extracted_skills:
                skill_filters = list(set((skill_filters or []) + extracted_skills))

        query_skills = [s.strip() for s in query.split(",") if s.strip()]
        if skill_filters:
            query_skills = list(set(query_skills + skill_filters))

        lexical_scores = self._score_fallback_lexical(query, query_skills)
        if self.model is not None and self.fallback_embeddings is not None:
            query_embedding = self.model.encode([query]).astype("float32")
            query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_norm[query_norm == 0] = 1.0
            query_embedding = query_embedding / query_norm
            cosine_scores = np.clip(np.dot(self.fallback_embeddings, query_embedding[0]), 0.0, 1.0)
            base_scores = (0.75 * cosine_scores) + (0.25 * lexical_scores)
        else:
            base_scores = lexical_scores

        target_companies = ["amazon", "google", "meta", "microsoft", "apple", "netflix"]
        jd_companies = [c for c in target_companies if c in query.lower()]

        scored = []
        for meta, score in zip(self.metadata, base_scores):
            text = meta.get("text", "")
            member_info = self._lookup_member(meta.get("filename", ""), text)
            grad_year = member_info.get("graduation_year", meta.get("graduation_year", ""))
            major = member_info.get("major", meta.get("major", ""))

            if grad_year_filter and grad_year != grad_year_filter:
                continue
            if major_filter and major_filter.lower() not in str(major).lower():
                continue

            matched = extract_matched_skills(text, query_skills) if query_skills else []
            final_score = float(score)
            if query_skills and matched:
                match_ratio = len(matched) / len(query_skills)
                final_score += 0.2 * match_ratio

                for company in jd_companies:
                    is_match = company in text.lower() or (company == "amazon" and "aws" in text.lower())
                    if is_match:
                        final_score += 0.18

            if final_score < min_score:
                continue

            scored.append(
                ResumeResult(
                    rank=0,
                    filename=meta.get("filename", ""),
                    score=min(final_score, RECRUITER_MAX_SCORE),
                    semantic_score=float(score),
                    file_path=meta.get("file_path", ""),
                    local_resume_path=self._resolve_resume_path(
                        meta.get("filename", ""),
                        meta.get("file_path", ""),
                    ),
                    text_preview=text[:400],
                    full_text=text,
                    source=meta.get("source", "ds3_members"),
                    full_name=member_info.get("full_name", meta.get("full_name", meta.get("filename", ""))),
                    major=major,
                    graduation_year=str(grad_year),
                    resume_link=member_info.get("resume_link", meta.get("resume_link", "")),
                    linkedin=member_info.get("linkedin", meta.get("linkedin", "")),
                    github=member_info.get("github", meta.get("github", "")),
                    matched_skills=matched,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        if input_mode == "Job Description":
            scored = self._apply_recruiter_reranking(scored, query, api_key)
        scored = scored[:top_k]

        for i, result in enumerate(scored, 1):
            result.rank = i

        return scored

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
                    semantic_score=min(score, 1.0),
                    file_path=meta.get("file_path", ""),
                    local_resume_path=self._resolve_resume_path(
                        meta.get("filename", ""),
                        meta.get("file_path", ""),
                    ),
                    text_preview=meta.get("text", "")[:400],
                    full_text=meta.get("text", ""),
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

    def _apply_recruiter_reranking(
        self,
        results: list[ResumeResult],
        job_description: str,
        api_key: str | None = None,
    ) -> list[ResumeResult]:
        rerank_count = min(RECRUITER_RERANK_LIMIT, len(results))
        if rerank_count == 0:
            return results

        for result in results[:rerank_count]:
            recruiter_assessment = assess_candidate_with_grok(
                job_description=job_description,
                candidate_text=result.full_text or result.text_preview,
                candidate_name=result.full_name or result.filename,
                api_key=api_key,
            )
            rubric_assessment = assess_resume_quality_with_grok(
                job_description=job_description,
                candidate_text=result.full_text or result.text_preview,
                candidate_name=result.full_name or result.filename,
                api_key=api_key,
            )

            recruiter_signal = (
                (recruiter_assessment.get("impact_score", 0.0) * 0.35)
                + (recruiter_assessment.get("technology_fit_score", 0.0) * 0.35)
                + (recruiter_assessment.get("keyword_alignment_score", 0.0) * 0.20)
                + (recruiter_assessment.get("role_fit_score", 0.0) * 0.10)
            ) / 10.0
            quality_signal = (
                (rubric_assessment.get("ats_format_score", 0.0) * 0.18)
                + (rubric_assessment.get("section_quality_score", 0.0) * 0.12)
                + (rubric_assessment.get("bullet_quality_score", 0.0) * 0.20)
                + (rubric_assessment.get("technical_relevance_score", 0.0) * 0.22)
                + (rubric_assessment.get("truthfulness_score", 0.0) * 0.18)
                + (rubric_assessment.get("project_strength_score", 0.0) * 0.10)
            ) / 10.0
            base_signal = min(result.score / RECRUITER_MAX_SCORE, 1.0)
            hard_fail_flags = rubric_assessment.get("hard_fail_flags", [])
            revision_flags = rubric_assessment.get("revision_flags", [])
            penalty = min(
                0.45,
                (0.12 * len(hard_fail_flags)) + (0.04 * len(revision_flags)),
            )
            blended_signal = max(
                0.0,
                (
                    (RECRUITER_RERANK_BASE_WEIGHT * base_signal)
                    + (RECRUITER_RERANK_MATCH_WEIGHT * recruiter_signal)
                    + (RECRUITER_RERANK_QUALITY_WEIGHT * quality_signal)
                    - penalty
                ),
            )

            result.recruiter_score = round(recruiter_signal * 10.0, 2)
            result.recruiter_breakdown = recruiter_assessment
            result.resume_quality_score = round(quality_signal * 10.0, 2)
            result.resume_quality_breakdown = rubric_assessment
            result.resume_flags = revision_flags
            result.hard_fail_flags = hard_fail_flags
            result.score = round(
                min(blended_signal * RECRUITER_MAX_SCORE, RECRUITER_MAX_SCORE),
                4,
            )

        results.sort(key=lambda item: item.score, reverse=True)
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

    def _resolve_resume_path(self, filename: str, file_path: str = "") -> str:
        candidates: list[Path] = []
        if file_path:
            candidates.append(Path(file_path))

        if filename:
            for base_dir in LOCAL_MEMBER_RESUME_DIRS:
                candidates.append(base_dir / filename)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

        return ""

    def _prepare_fallback_text_index(self):
        self.fallback_term_frequencies = []
        self.fallback_doc_lengths = []
        self.fallback_idf = {}
        self.fallback_avg_doc_length = 0.0

        if not self.metadata:
            return

        document_frequency: dict[str, int] = {}
        for meta in self.metadata:
            tokens = _tokenize_text(meta.get("text", ""))
            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            self.fallback_term_frequencies.append(term_counts)
            self.fallback_doc_lengths.append(len(tokens))
            for token in term_counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        num_docs = len(self.metadata)
        self.fallback_avg_doc_length = sum(self.fallback_doc_lengths) / max(num_docs, 1)
        for token, doc_freq in document_frequency.items():
            self.fallback_idf[token] = np.log(1 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def _score_fallback_lexical(self, query: str, query_skills: list[str]) -> np.ndarray:
        if not self.metadata:
            return np.array([], dtype="float32")

        query_tokens = _tokenize_text(query)
        if query_skills:
            for skill in query_skills:
                query_tokens.extend(_tokenize_text(skill))
        query_terms = list(dict.fromkeys(query_tokens))

        if not query_terms:
            return np.zeros(len(self.metadata), dtype="float32")

        scores = np.zeros(len(self.metadata), dtype="float32")
        k1 = 1.5
        b = 0.75

        for i, (meta, term_counts, doc_length) in enumerate(
            zip(self.metadata, self.fallback_term_frequencies, self.fallback_doc_lengths)
        ):
            score = 0.0
            for term in query_terms:
                freq = term_counts.get(term, 0)
                if freq == 0:
                    continue
                idf = self.fallback_idf.get(term, 0.0)
                denom = freq + k1 * (
                    1 - b + b * doc_length / max(self.fallback_avg_doc_length, 1.0)
                )
                score += idf * ((freq * (k1 + 1)) / max(denom, 1e-6))

            lower_text = meta.get("text", "").lower()
            for skill in query_skills:
                if skill and skill.lower() in lower_text:
                    score += 0.75

            scores[i] = score

        max_score = float(scores.max()) if len(scores) else 0.0
        if max_score > 0:
            scores = scores / max_score
        return scores

    def _load_fallback_embedding_map(self) -> dict[str, np.ndarray] | None:
        if not FALLBACK_EMBEDDINGS_PATH.exists() or not FALLBACK_EMBEDDING_METADATA_PATH.exists():
            return None

        try:
            embeddings = np.load(FALLBACK_EMBEDDINGS_PATH)
            with open(FALLBACK_EMBEDDING_METADATA_PATH, "r") as f:
                rows = json.load(f)
        except Exception as exc:
            print(f"[SearchEngine] Could not load fallback embeddings: {exc}")
            return None

        if len(embeddings) != len(rows):
            print("[SearchEngine] Fallback embeddings do not align with metadata rows.")
            return None

        embedding_map: dict[str, np.ndarray] = {}
        for row, embedding in zip(rows, embeddings):
            if row.get("source") != "ds3_members":
                continue
            filename = row.get("filename", "")
            if filename:
                embedding_map[filename] = np.asarray(embedding, dtype="float32")

        return embedding_map or None

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
