from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from config import (
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
        RERANKER_ENABLED,
        RERANKER_MODEL_PATH,
    )
    from job_description import ParsedJobDescription, parse_job_description
    from grok_utils import assess_candidate_packet_with_grok, has_grok_api_key
except ImportError:
    from streamlit.config import (
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
        RERANKER_ENABLED,
        RERANKER_MODEL_PATH,
    )
    from streamlit.job_description import ParsedJobDescription, parse_job_description
    from streamlit.grok_utils import assess_candidate_packet_with_grok, has_grok_api_key


PROCESSED_DIR = DATA_DIR / "processed"
PARSED_RESUMES_PATH = PROCESSED_DIR / "resumes_parsed.json"
RESUME_CHUNKS_PATH = PROCESSED_DIR / "resume_chunks.json"
CHUNK_METADATA_PATH = PROJECT_ROOT / "member_chunks_metadata.json"
LOCAL_MEMBER_RESUME_DIRS = [
    MEMBER_RESUMES_DIR,
    PROJECT_ROOT / "test" / "members",
]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "if",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "will", "with", "you", "your",
}
SECTION_WEIGHTS = {
    "experience": 1.0,
    "projects": 0.9,
    "skills": 0.85,
    "education": 0.55,
    "summary": 0.45,
    "contact": 0.2,
}
_SKILL_PATTERNS: dict[str, re.Pattern] = {}
COMPANY_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "plc",
    "lp",
}
COMPANY_LOCATION_RE = re.compile(
    r"\b(?:remote|hybrid|on[- ]site|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})\b.*$"
)
FINAL_PAGE_PENALTY = 0.04
GROK_TOP_N = 10
GROK_MAX_WORKERS = max(1, min(8, int(os.getenv("TALENTLENS_GROK_MAX_WORKERS", "6"))))
STRONG_COMPANY_BOOST = 0.18
WEAK_COMPANY_MENTION_BOOST = 0.05


@dataclass
class ResumeResult:
    rank: int
    filename: str
    score: float
    semantic_score: float
    file_path: str
    local_resume_path: str
    text_preview: str
    candidate_id: str = ""
    full_text: str = ""
    source: str = ""
    full_name: str = ""
    major: str = ""
    graduation_year: str = ""
    resume_link: str = ""
    linkedin: str = ""
    github: str = ""
    matched_skills: list[str] = field(default_factory=list)
    explanation: str = ""
    recruiter_score: float = 0.0
    recruiter_breakdown: dict = field(default_factory=dict)
    resume_quality_score: float = 0.0
    resume_quality_breakdown: dict = field(default_factory=dict)
    resume_flags: list[str] = field(default_factory=list)
    hard_fail_flags: list[str] = field(default_factory=list)
    ranking_details: dict = field(default_factory=dict)
    top_evidence_chunks: list[dict] = field(default_factory=list)
    hard_filter_status: dict = field(default_factory=dict)
    reranker_score: float = 0.0
    retrieval_score: float = 0.0
    must_have_coverage: float = 0.0
    page_count: int | None = None
    company_match_status: str = "not_requested"
    grok_status: str = "not_requested"
    grok_fit_score: float = 0.0
    grok_resume_quality_score: float = 0.0
    grok_summary: str = ""
    grok_matched_requirements: list[str] = field(default_factory=list)
    grok_missing_requirements: list[str] = field(default_factory=list)
    grok_weakness_flags: list[str] = field(default_factory=list)


@dataclass
class ChunkHit:
    candidate_id: str
    chunk_id: str
    section_type: str
    text: str
    score: float
    source: str


def _build_skill_patterns():
    if _SKILL_PATTERNS:
        return
    for skill in SKILL_SUGGESTIONS:
        escaped = re.escape(skill)
        _SKILL_PATTERNS[skill] = re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def extract_matched_skills(text: str, query_skills: list[str]) -> list[str]:
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


def _normalize_company_name(name: str) -> str:
    cleaned = (name or "").lower().replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    tokens = cleaned.split()
    while tokens and tokens[-1] in COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _normalize_free_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _company_names_equivalent(left: str, right: str) -> bool:
    normalized_left = _normalize_company_name(left)
    normalized_right = _normalize_company_name(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True

    left_tokens = normalized_left.split()
    right_tokens = normalized_right.split()
    if len(left_tokens) == 1 and normalized_left in normalized_right:
        return True
    if len(right_tokens) == 1 and normalized_right in normalized_left:
        return True
    return False


def _extract_employer_names(experience_entries: list[dict]) -> list[str]:
    employers: list[str] = []
    for entry in experience_entries or []:
        if not isinstance(entry, dict):
            continue
        raw_text = str(entry.get("raw_text", "")).strip()
        if not raw_text:
            continue
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        company_line = lines[1]
        parts = re.split(r"\s{2,}", company_line, maxsplit=1)
        employer = parts[0].strip()
        if len(parts) == 1:
            employer = re.sub(
                r"\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}|Remote|Hybrid|On[- ]site)$",
                "",
                employer,
            ).strip(" -|,")
        if employer:
            employers.append(employer)
    return employers


ProgressCallback = Callable[[float, str], None]


class SearchEngine:
    def __init__(self):
        self.model = None
        self.index = None
        self.resume_metadata: list[dict] = []
        self.resume_metadata_by_filename: dict[str, dict] = {}
        self.chunk_metadata: list[dict] = []
        self.chunk_candidates: list[dict] = []
        self.chunk_term_frequencies: list[dict[str, int]] = []
        self.chunk_doc_lengths: list[int] = []
        self.chunk_idf: dict[str, float] = {}
        self.chunk_avg_doc_length: float = 0.0
        self.parsed_resume_map: dict[str, dict] = {}
        self.resume_term_frequencies: list[dict[str, int]] = []
        self.resume_doc_lengths: list[int] = []
        self.resume_idf: dict[str, float] = {}
        self.resume_avg_doc_length: float = 0.0
        self.members_df: Optional[pd.DataFrame] = None
        self._member_index: dict[str, dict] = {}
        self.last_query_analysis: dict | None = None
        self.demo_mode = False
        self.mode_label = "Live"
        self.mode_banner = ""
        self.retrieval_backend = "uninitialized"
        self.reranker = None
        self.reranker_loaded = False
        self._page_count_cache: dict[str, int | None] = {}
        self._load()

    def _load(self):
        if MEMBERS_CSV.exists():
            self.members_df = pd.read_csv(MEMBERS_CSV)
            self._build_member_index()

        self._load_parsed_resumes()
        self._load_chunk_records()
        self._load_resume_metadata()
        self._prepare_resume_text_index()
        self._load_semantic_backend()
        self._load_reranker()

        if self.chunk_metadata:
            chunk_msg = f"Chunk retrieval ready ({self.retrieval_backend})."
        elif self.resume_metadata:
            chunk_msg = "Chunk artifacts unavailable, falling back to resume-level retrieval."
        else:
            chunk_msg = "Resume artifacts unavailable, using demo data."

        if not self.chunk_metadata and not self.resume_metadata:
            self._load_demo()
            return

        self.demo_mode = False
        self.mode_label = "Live" if self.retrieval_backend.startswith("semantic") else "Fallback"
        self.mode_banner = chunk_msg

    def _load_parsed_resumes(self):
        if not PARSED_RESUMES_PATH.exists():
            return
        with open(PARSED_RESUMES_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)

        for row in rows:
            if row.get("source") != "ds3_members":
                continue
            candidate_id = row.get("candidate_id", "")
            if candidate_id:
                self.parsed_resume_map[candidate_id] = row

    def _load_chunk_records(self):
        if not RESUME_CHUNKS_PATH.exists():
            return
        with open(RESUME_CHUNKS_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)

        self.chunk_candidates = [row for row in rows if row.get("source") == "ds3_members"]
        self._prepare_chunk_text_index()

    def _load_resume_metadata(self):
        if not METADATA_PATH.exists():
            return
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)
        self.resume_metadata = [row for row in rows if row.get("source") == "ds3_members"]
        self.resume_metadata_by_filename = {
            row.get("filename", ""): row for row in self.resume_metadata if row.get("filename")
        }

    def _load_semantic_backend(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except Exception:
            self.retrieval_backend = "lexical-chunk"
            return

        if not FAISS_INDEX_PATH.exists():
            self.retrieval_backend = "lexical-chunk"
            return

        metadata_rows = None
        metadata_kind = None
        if CHUNK_METADATA_PATH.exists():
            with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata_rows = json.load(f)
            metadata_kind = "chunk"
        elif METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata_rows = json.load(f)
            metadata_kind = "resume"

        if metadata_rows is None:
            self.retrieval_backend = "lexical-chunk"
            return

        try:
            model = SentenceTransformer(MODEL_NAME, local_files_only=True)
            index = faiss.read_index(str(FAISS_INDEX_PATH))
        except Exception:
            self.retrieval_backend = "lexical-chunk"
            return

        if len(metadata_rows) != index.ntotal:
            self.retrieval_backend = "lexical-chunk"
            return

        self.model = model
        self.index = index
        if metadata_kind == "chunk":
            self.chunk_metadata = metadata_rows
            self.retrieval_backend = "semantic-chunk"
        else:
            self.retrieval_backend = "semantic-resume"

    def _load_reranker(self):
        """Lazily load the cross-encoder reranker if enabled and available."""
        if self.reranker_loaded:
            return
        if not RERANKER_ENABLED:
            self.reranker = None
            self.reranker_loaded = False
            return
        if not RERANKER_MODEL_PATH.exists():
            self.reranker = None
            self.reranker_loaded = False
            return
        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(
                str(RERANKER_MODEL_PATH),
                max_length=512,
                activation_fn=None,
            )
            self.reranker_loaded = True
        except Exception:
            self.reranker = None
            self.reranker_loaded = False

    def _load_demo(self):
        self.demo_mode = True
        self.mode_label = "Demo"
        self.mode_banner = "No retrieval artifacts found. Showing synthetic demo data."
        demo_names = [
            ("Alice Chen", "Computer Science (B.S.)", "2026"),
            ("Bob Patel", "Data Science (B.S.)", "2027"),
            ("Carol Kim", "Computer Engineering (B.S.)", "2026"),
            ("David Lopez", "Mathematics (B.S.)", "2028"),
        ]
        skills_pool = [
            "Python, Machine Learning, TensorFlow, SQL, Pandas",
            "Java, React, Node.js, AWS, Docker",
            "C++, Computer Vision, PyTorch, CUDA, OpenCV",
            "R, Statistics, Tableau, Power BI, Excel",
        ]
        self.resume_metadata = []
        for (name, major, grad_year), skills in zip(demo_names, skills_pool):
            filename = f"{name.replace(' ', '_').lower()}_resume.pdf"
            self.resume_metadata.append(
                {
                    "filename": filename,
                    "file_path": f"data/ds3/member_resumes/{filename}",
                    "text": f"{name}\n{major}\nSkills: {skills}",
                    "source": "demo",
                    "full_name": name,
                    "major": major,
                    "graduation_year": grad_year,
                }
            )
        self.resume_metadata_by_filename = {row["filename"]: row for row in self.resume_metadata}

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
        progress_callback: ProgressCallback | None = None,
    ) -> list[ResumeResult]:
        self.last_query_analysis = None
        if self.demo_mode:
            return self._search_demo(query, top_k, skill_filters, grad_year_filter, major_filter)
        if input_mode == "Job Description":
            return self._search_job_description(
                query,
                top_k,
                min_score,
                grad_year_filter,
                major_filter,
                api_key,
                progress_callback,
            )
        self._emit_progress(progress_callback, 0.1, "Matching skill filters")
        results = self._search_skills(query, top_k, min_score, skill_filters, grad_year_filter, major_filter)
        self._emit_progress(progress_callback, 1.0, "Search complete")
        return results

    def _search_job_description(
        self,
        query: str,
        top_k: int,
        min_score: float,
        grad_year_filter: str | None,
        major_filter: str | None,
        api_key: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[ResumeResult]:
        self._emit_progress(progress_callback, 0.05, "Parsing job description")
        parsed = parse_job_description(query)
        chunk_fetch_k = max(top_k * 10, 250)
        candidate_pool_limit = max(top_k * 3, 30)
        self._emit_progress(progress_callback, 0.15, "Retrieving resume evidence")
        chunk_hits = self._retrieve_chunks(query, parsed, top_k=chunk_fetch_k)
        self._emit_progress(progress_callback, 0.35, "Aggregating candidate pool")
        results = self._aggregate_chunk_hits(
            chunk_hits=chunk_hits,
            parsed=parsed,
            top_k=candidate_pool_limit,
            min_score=min_score,
            grad_year_filter=grad_year_filter,
            major_filter=major_filter,
        )
        self.last_query_analysis = {
            **parsed.to_dict(),
            "retrieval": {
                "backend": self.retrieval_backend,
                "top_k_chunks": chunk_fetch_k,
                "retrieved_chunks": len(chunk_hits),
                "shortlisted_candidates": len(results),
                "candidate_pool_limit": candidate_pool_limit,
                "reranker_active": self.reranker_loaded,
                "grok_top_n": GROK_TOP_N,
            },
        }
        if self.reranker_loaded:
            self._emit_progress(progress_callback, 0.55, "Reranking shortlisted candidates")
            results = self._rerank(query, results, parsed)
        self._emit_progress(progress_callback, 0.65, "Evaluating top candidates with Grok")
        results = self._apply_grok_scores(
            query,
            results,
            parsed,
            top_n=GROK_TOP_N,
            api_key=api_key,
            final_top_k=top_k,
            progress_callback=progress_callback,
        )
        self._emit_progress(progress_callback, 1.0, "Search complete")
        return results[:top_k]

    def _search_skills(
        self,
        query: str,
        top_k: int,
        min_score: float,
        skill_filters: list[str] | None,
        grad_year_filter: str | None,
        major_filter: str | None,
    ) -> list[ResumeResult]:
        query_skills = [s.strip() for s in query.split(",") if s.strip()]
        if skill_filters:
            query_skills = list(dict.fromkeys(query_skills + skill_filters))

        scored_rows: list[ResumeResult] = []
        scores = self._resume_scores(query, query_skills)
        for meta, raw_score in zip(self.resume_metadata, scores):
            if raw_score < min_score:
                continue
            member_info = self._lookup_member(meta.get("filename", ""), meta.get("text", ""))
            grad_year = member_info.get("graduation_year", meta.get("graduation_year", ""))
            major = member_info.get("major", meta.get("major", ""))
            if grad_year_filter and str(grad_year) != grad_year_filter:
                continue
            if major_filter and major_filter.lower() not in str(major).lower():
                continue
            text = meta.get("text", "")
            matched = extract_matched_skills(text, query_skills) if query_skills else []
            final_score = min(1.0, float(raw_score) + (0.12 * len(matched) / max(len(query_skills), 1)))
            scored_rows.append(
                ResumeResult(
                    rank=0,
                    filename=meta.get("filename", ""),
                    candidate_id=meta.get("filename", ""),
                    score=final_score,
                    semantic_score=float(raw_score),
                    file_path=meta.get("file_path", ""),
                    local_resume_path=self._resolve_resume_path(meta.get("filename", ""), meta.get("file_path", "")),
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
                    retrieval_score=final_score,
                    top_evidence_chunks=[
                        {
                            "section_type": "resume",
                            "score": round(float(raw_score), 4),
                            "text": text[:300],
                        }
                    ],
                    ranking_details={
                        "mode": "skill_search",
                        "retrieval_backend": self.retrieval_backend,
                        "base_search_score": round(float(raw_score), 4),
                        "matched_skill_count": len(matched),
                    },
                )
            )

        scored_rows.sort(key=lambda item: item.score, reverse=True)
        top_results = scored_rows[:top_k]

        if self.reranker_loaded and self.reranker is not None and top_results:
            top_results = self._rerank(query, top_results)

        for i, result in enumerate(top_results, 1):
            result.rank = i
        return top_results

    def _rerank(self, query: str, results: list[ResumeResult], parsed: ParsedJobDescription | None = None) -> list[ResumeResult]:
        """Second-stage reranking with a fine-tuned CrossEncoder."""
        if not self.reranker_loaded or self.reranker is None or not results:
            return results

        pairs: list[tuple[str, str]] = []
        must_haves = []
        parsed_company = ""

        if parsed:
            must_haves = parsed.must_have_skills
            parsed_company = parsed.company
        else:
            must_haves = (getattr(results[0], "hard_filter_status", {}) or {}).get("matched_must_have_skills", []) or []
            parsed_company = (getattr(self, "last_query_analysis", {}) or {}).get("company", "")

        prefix = f"{parsed_company} " if parsed_company else ""
        clean_query = f"{prefix}{query.splitlines()[0]} | {', '.join(must_haves)}"

        for r in results:
            evidence_text = ""
            if r.top_evidence_chunks:
                evidence_text = "\n".join(str(h.get("text", "")).strip() for h in r.top_evidence_chunks[:5])
            if not evidence_text:
                evidence_text = r.full_text or r.text_preview or ""
            pairs.append((clean_query, evidence_text[:2000]))

        try:
            raw_scores = self.reranker.predict(pairs)
            scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))
        except Exception:
            return results

        for r, s in zip(results, scores):
            rerank_score = float(s)
            r.reranker_score = rerank_score
            r.ranking_details["reranker_score"] = round(rerank_score, 4)
            r.ranking_details["reranker_model"] = "talentlens-cross-encoder-sft-v1"

            retrieval = float(r.retrieval_score or r.ranking_details.get("retrieval_score", r.score))
            coverage = float(r.must_have_coverage)
            if parsed is None:
                pre_grok_score = (0.80 * retrieval) + (0.20 * rerank_score)
            else:
                pre_grok_score = (0.65 * retrieval) + (0.20 * rerank_score) + (0.15 * coverage)

            r.score = max(0.001, min(0.999, pre_grok_score))
            r.ranking_details["pre_grok_score"] = round(r.score, 4)
            r.ranking_details["pre_grok_components"] = {
                "retrieval": round(retrieval, 4),
                "reranker": round(rerank_score, 4),
                "must_have_coverage": round(coverage, 4),
            }

        results.sort(key=lambda item: item.score, reverse=True)
        for i, result in enumerate(results, 1):
            result.rank = i
        return results

    def _get_resume_page_count(self, local_resume_path: str) -> int | None:
        if not local_resume_path:
            return None
        if local_resume_path in self._page_count_cache:
            return self._page_count_cache[local_resume_path]
        try:
            try:
                import fitz
            except ImportError:
                import pymupdf as fitz

            with fitz.open(local_resume_path) as doc:
                page_count = int(doc.page_count)
        except Exception:
            page_count = None
        self._page_count_cache[local_resume_path] = page_count
        return page_count

    def _get_company_match_signal(self, parsed_company: str, profile: dict[str, Any]) -> tuple[str, float]:
        normalized_company = _normalize_company_name(parsed_company)
        if not normalized_company:
            return "not_requested", 0.0

        for employer in profile.get("employer_names", []):
            if _company_names_equivalent(parsed_company, employer):
                return "exact_experience_match", STRONG_COMPANY_BOOST

        normalized_text = _normalize_free_text(profile.get("combined_text", ""))
        if normalized_company and normalized_company in normalized_text:
            return "loose_mention", WEAK_COMPANY_MENTION_BOOST
        return "no_match", 0.0

    def _build_candidate_packet(
        self,
        result: ResumeResult,
        profile: dict[str, Any],
        parsed: ParsedJobDescription,
    ) -> dict[str, Any]:
        education_entries = []
        for entry in profile.get("education_entries", [])[:2]:
            if isinstance(entry, dict) and entry.get("raw_text"):
                education_entries.append(str(entry["raw_text"])[:350])

        experience_entries = []
        for entry in profile.get("experience_entries", [])[:2]:
            if isinstance(entry, dict) and entry.get("raw_text"):
                experience_entries.append(str(entry["raw_text"])[:500])

        project_entries = []
        for entry in profile.get("project_entries", [])[:2]:
            if isinstance(entry, dict) and entry.get("raw_text"):
                project_entries.append(str(entry["raw_text"])[:500])

        skills = profile.get("skills", [])
        if not isinstance(skills, list):
            skills = [str(skills)]

        return {
            "candidate_id": result.candidate_id,
            "candidate_name": result.full_name or result.candidate_id,
            "major": result.major,
            "graduation_year": result.graduation_year,
            "summary": profile.get("summary", ""),
            "skills": skills[:25],
            "education": education_entries,
            "experience": experience_entries,
            "projects": project_entries,
            "estimated_years_experience": profile.get("estimated_years_experience"),
            "company_match_status": result.company_match_status,
            "page_count": result.page_count,
            "matched_skills": result.matched_skills,
            "matched_must_haves": result.hard_filter_status.get("matched_must_have_skills", []),
            "matched_preferred": result.hard_filter_status.get("matched_preferred_skills", []),
            "top_evidence_chunks": [
                {
                    "section_type": chunk.get("section_type", "other"),
                    "score": chunk.get("score", 0.0),
                    "text": str(chunk.get("text", ""))[:350],
                }
                for chunk in result.top_evidence_chunks[:2]
            ],
            "job_company": parsed.company,
        }

    def _emit_progress(
        self,
        progress_callback: ProgressCallback | None,
        progress: float,
        message: str,
    ) -> None:
        if callable(progress_callback):
            progress_callback(max(0.0, min(1.0, progress)), message)

    def _apply_grok_scores(
        self,
        query: str,
        results: list[ResumeResult],
        parsed: ParsedJobDescription,
        top_n: int,
        api_key: str | None = None,
        final_top_k: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[ResumeResult]:
        if not results:
            return results

        parsed_requirements = {
            key: value for key, value in parsed.to_dict().items() if key != "raw_text"
        }
        top_n = min(top_n, len(results))
        page_count_limit = min(len(results), max(top_n, final_top_k or 0))
        grok_available = has_grok_api_key(api_key)

        for index, result in enumerate(results):
            if index < page_count_limit:
                result.page_count = self._get_resume_page_count(result.local_resume_path)
            if index >= top_n:
                result.grok_status = "skipped"
        tasks: list[tuple[ResumeResult, dict[str, Any]]] = []
        if grok_available:
            for result in results[:top_n]:
                profile = self._get_candidate_profile(result.candidate_id)
                tasks.append((result, self._build_candidate_packet(result, profile, parsed)))
        else:
            for result in results[:top_n]:
                result.grok_status = "unavailable"

        def score_task(task: tuple[ResumeResult, dict[str, Any]]) -> tuple[ResumeResult, dict[str, Any]]:
            result, candidate_packet = task
            assessment = assess_candidate_packet_with_grok(
                job_description=query,
                parsed_requirements=parsed_requirements,
                candidate_packet=candidate_packet,
                api_key=api_key,
            )
            return result, assessment

        scored_assessments: list[tuple[ResumeResult, dict[str, Any]]] = []
        if tasks:
            max_workers = min(GROK_MAX_WORKERS, len(tasks))
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {executor.submit(score_task, task): task[0] for task in tasks}
                    for completed_count, future in enumerate(as_completed(future_map), 1):
                        scored_assessments.append(future.result())
                        progress = 0.65 + (0.30 * completed_count / max(len(tasks), 1))
                        self._emit_progress(
                            progress_callback,
                            progress,
                            f"Evaluating top candidates with Grok ({completed_count}/{len(tasks)})",
                        )
            else:
                for completed_count, task in enumerate(tasks, 1):
                    scored_assessments.append(score_task(task))
                    progress = 0.65 + (0.30 * completed_count / max(len(tasks), 1))
                    self._emit_progress(
                        progress_callback,
                        progress,
                        f"Evaluating top candidates with Grok ({completed_count}/{len(tasks)})",
                    )

        for result, assessment in scored_assessments:
            result.grok_status = str(assessment.get("status", "error"))
            qualification_score = float(assessment.get("qualification_match_score", 0.0))
            company_score = float(assessment.get("company_relevance_score", 0.0))
            experience_score = float(assessment.get("experience_relevance_score", 0.0))
            bullet_score = float(assessment.get("bullet_quality_score", 0.0))
            project_score = float(assessment.get("project_strength_score", 0.0))
            resume_score = float(assessment.get("resume_quality_score", 0.0))

            grok_fit_raw = (qualification_score + company_score + experience_score) / 3.0
            grok_quality_raw = (bullet_score + project_score + resume_score) / 3.0

            result.grok_fit_score = grok_fit_raw / 10.0
            result.grok_resume_quality_score = grok_quality_raw / 10.0
            result.grok_summary = str(assessment.get("summary", "")).strip()
            result.grok_matched_requirements = list(assessment.get("matched_requirements", []))
            result.grok_missing_requirements = list(assessment.get("missing_requirements", []))
            result.grok_weakness_flags = list(assessment.get("weakness_flags", []))
            result.recruiter_score = result.grok_fit_score
            result.resume_quality_score = result.grok_resume_quality_score
            result.recruiter_breakdown = {
                "qualification_match_score": qualification_score,
                "company_relevance_score": company_score,
                "experience_relevance_score": experience_score,
            }
            result.resume_quality_breakdown = {
                "bullet_quality_score": bullet_score,
                "project_strength_score": project_score,
                "resume_quality_score": resume_score,
                "status": result.grok_status,
            }

        for result in results:
            final_score = self._compute_final_score(result)
            result.score = final_score
            result.ranking_details["final_score_components"] = {
                "retrieval": round(result.retrieval_score, 4),
                "reranker": round(result.reranker_score, 4),
                "must_have_coverage": round(result.must_have_coverage, 4),
                "grok_fit": round(result.grok_fit_score, 4),
                "grok_quality": round(result.grok_resume_quality_score, 4),
                "page_penalty": round(FINAL_PAGE_PENALTY if (result.page_count or 0) > 1 else 0.0, 4),
            }
            result.ranking_details["grok_status"] = result.grok_status
            result.ranking_details["company_match_status"] = result.company_match_status
            result.ranking_details["page_count"] = result.page_count
        self._emit_progress(progress_callback, 0.97, "Finalizing ranking")

        results.sort(key=lambda item: item.score, reverse=True)
        for i, result in enumerate(results, 1):
            result.rank = i
        return results

    def _compute_final_score(self, result: ResumeResult) -> float:
        page_penalty = FINAL_PAGE_PENALTY if (result.page_count or 0) > 1 else 0.0
        final_score = (
            (0.30 * float(result.retrieval_score))
            + (0.20 * float(result.reranker_score))
            + (0.15 * float(result.must_have_coverage))
            + (0.20 * float(result.grok_fit_score))
            + (0.15 * float(result.grok_resume_quality_score))
            - page_penalty
        )
        return max(0.001, min(0.999, final_score))

    def _retrieve_chunks(self, query: str, parsed: ParsedJobDescription, top_k: int) -> list[ChunkHit]:
        if self.retrieval_backend == "semantic-chunk" and self.index is not None and self.model is not None and self.chunk_metadata:
            return self._semantic_chunk_search(query, top_k)
        if self.chunk_candidates:
            return self._lexical_chunk_search(query, parsed, top_k)
        return self._resume_as_chunk_search(query, top_k)

    def _semantic_chunk_search(self, query: str, top_k: int) -> list[ChunkHit]:
        import faiss as _faiss

        query_embedding = self.model.encode([query], normalize_embeddings=True).astype("float32")
        _faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        n_meta = len(self.chunk_metadata)
        hits: list[ChunkHit] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= n_meta:
                continue
            row = self.chunk_metadata[idx]
            hits.append(
                ChunkHit(
                    candidate_id=row.get("candidate_id", ""),
                    chunk_id=row.get("chunk_id", ""),
                    section_type=row.get("section_type", "other"),
                    text=row.get("text", ""),
                    score=float(max(0.0, score)) * SECTION_WEIGHTS.get(row.get("section_type", "other"), 0.65),
                    source=row.get("source", "ds3_members"),
                )
            )
        return hits

    def _lexical_chunk_search(self, query: str, parsed: ParsedJobDescription, top_k: int) -> list[ChunkHit]:
        query_skills = list(dict.fromkeys(parsed.must_have_skills + parsed.preferred_skills))
        scores = self._score_bm25(
            docs=self.chunk_candidates,
            term_frequencies=self.chunk_term_frequencies,
            doc_lengths=self.chunk_doc_lengths,
            avg_doc_length=self.chunk_avg_doc_length,
            idf_map=self.chunk_idf,
            query=query,
            query_skills=query_skills,
        )
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        hits: list[ChunkHit] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            row = self.chunk_candidates[int(idx)]
            hits.append(
                ChunkHit(
                    candidate_id=row.get("candidate_id", ""),
                    chunk_id=row.get("chunk_id", ""),
                    section_type=row.get("section_type", "other"),
                    text=row.get("text", ""),
                    score=score * SECTION_WEIGHTS.get(row.get("section_type", "other"), 0.65),
                    source=row.get("source", "ds3_members"),
                )
            )
        return hits

    def _resume_as_chunk_search(self, query: str, top_k: int) -> list[ChunkHit]:
        scores = self._resume_scores(query, [])
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        hits: list[ChunkHit] = []
        for idx in ranked_indices:
            row = self.resume_metadata[int(idx)]
            hits.append(
                ChunkHit(
                    candidate_id=row.get("filename", ""),
                    chunk_id=row.get("filename", ""),
                    section_type="resume",
                    text=row.get("text", "")[:500],
                    score=float(scores[idx]),
                    source=row.get("source", "ds3_members"),
                )
            )
        return hits

    def _aggregate_chunk_hits(
        self,
        chunk_hits: list[ChunkHit],
        parsed: ParsedJobDescription,
        top_k: int,
        min_score: float,
        grad_year_filter: str | None,
        major_filter: str | None,
    ) -> list[ResumeResult]:
        grouped: dict[str, list[ChunkHit]] = {}
        for hit in chunk_hits:
            if not hit.candidate_id:
                continue
            grouped.setdefault(hit.candidate_id, []).append(hit)

        results: list[ResumeResult] = []
        for candidate_id, hits in grouped.items():
            hits.sort(key=lambda item: item.score, reverse=True)
            profile = self._get_candidate_profile(candidate_id)
            if not profile:
                continue
            filter_status = self._evaluate_hard_filters(profile, parsed, grad_year_filter, major_filter)
            if not filter_status["passed"]:
                continue

            must_have_count = len(filter_status["matched_must_have_skills"])
            preferred_count = len(filter_status["matched_preferred_skills"])
            best_score = hits[0].score
            mean_top_scores = float(np.mean([hit.score for hit in hits[:3]]))
            must_have_ratio = 1.0
            if parsed.must_have_skills:
                must_have_ratio = must_have_count / max(len(parsed.must_have_skills), 1)
            preferred_ratio = 0.0
            if parsed.preferred_skills:
                preferred_ratio = preferred_count / max(len(parsed.preferred_skills), 1)

            company_match_status, company_boost = self._get_company_match_signal(parsed.company, profile)

            combined_score = min(
                1.0,
                (0.55 * best_score)
                + (0.10 * mean_top_scores)
                + (0.20 * must_have_ratio)
                + (0.05 * preferred_ratio)
                + company_boost,
            )
            if combined_score < max(min_score, 0.05):
                continue

            member_info = self._lookup_member(candidate_id, profile.get("combined_text", ""))
            display_meta = self.resume_metadata_by_filename.get(candidate_id, {})
            top_evidence = [
                {
                    "section_type": hit.section_type,
                    "score": round(hit.score, 4),
                    "text": hit.text,
                }
                for hit in hits[:3]
            ]
            matched_skills = sorted(
                set(filter_status["matched_must_have_skills"] + filter_status["matched_preferred_skills"])
            )

            results.append(
                ResumeResult(
                    rank=0,
                    filename=candidate_id,
                    candidate_id=candidate_id,
                    score=combined_score,
                    semantic_score=best_score,
                    file_path=display_meta.get("file_path", profile.get("file_path", "")),
                    local_resume_path=self._resolve_resume_path(candidate_id, display_meta.get("file_path", profile.get("file_path", ""))),
                    text_preview=hits[0].text[:400],
                    full_text=profile.get("combined_text", ""),
                    source=profile.get("source", "ds3_members"),
                    full_name=member_info.get("full_name", profile.get("full_name", candidate_id)),
                    major=member_info.get("major", profile.get("major", "")),
                    graduation_year=str(member_info.get("graduation_year", profile.get("graduation_year", ""))),
                    resume_link=member_info.get("resume_link", profile.get("resume_link", "")),
                    linkedin=member_info.get("linkedin", profile.get("linkedin", "")),
                    github=member_info.get("github", profile.get("github", "")),
                    matched_skills=matched_skills,
                    top_evidence_chunks=top_evidence,
                    hard_filter_status=filter_status,
                    retrieval_score=combined_score,
                    must_have_coverage=must_have_ratio,
                    company_match_status=company_match_status,
                    ranking_details={
                        "mode": "job_description_retrieval",
                        "retrieval_backend": self.retrieval_backend,
                        "base_search_score": round(best_score, 4),
                        "retrieval_score": round(combined_score, 4),
                        "mean_top_chunk_score": round(mean_top_scores, 4),
                        "evidence_chunk_count": len(hits),
                        "matched_must_have_count": must_have_count,
                        "total_must_have_count": len(parsed.must_have_skills),
                        "matched_preferred_count": preferred_count,
                        "total_preferred_count": len(parsed.preferred_skills),
                        "company_match_status": company_match_status,
                        "company_boost": round(company_boost, 4),
                    },
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        for i, result in enumerate(results[:top_k], 1):
            result.rank = i
        return results[:top_k]

    def _evaluate_hard_filters(
        self,
        profile: dict,
        parsed: ParsedJobDescription,
        grad_year_filter: str | None,
        major_filter: str | None,
    ) -> dict:
        combined_text = profile.get("combined_text", "")
        matched_must_have = extract_matched_skills(combined_text, parsed.must_have_skills)
        matched_preferred = extract_matched_skills(combined_text, parsed.preferred_skills)

        status = {
            "passed": True,
            "matched_must_have_skills": matched_must_have,
            "matched_preferred_skills": matched_preferred,
            "minimum_years_required": parsed.minimum_years_experience,
            "candidate_years_experience": profile.get("estimated_years_experience"),
            "years_experience_status": "not_requested",
            "degree_requirements": parsed.degree_requirements,
            "degree_status": "not_requested",
            "location": parsed.location,
            "remote_policy": parsed.remote_policy,
            "location_status": "not_requested",
        }

        if grad_year_filter and str(profile.get("graduation_year", "")) != grad_year_filter:
            status["passed"] = False
            status["graduation_year_status"] = "failed"
        else:
            status["graduation_year_status"] = "passed" if grad_year_filter else "not_requested"

        if major_filter and major_filter.lower() not in str(profile.get("major", "")).lower():
            status["passed"] = False
            status["major_status"] = "failed"
        else:
            status["major_status"] = "passed" if major_filter else "not_requested"

        if parsed.must_have_skills:
            if len(matched_must_have) < len(parsed.must_have_skills):
                status["must_have_status"] = "failed"
            else:
                status["must_have_status"] = "passed"
        else:
            status["must_have_status"] = "not_requested"

        candidate_years = profile.get("estimated_years_experience")
        if parsed.minimum_years_experience is not None:
            if candidate_years is None:
                status["years_experience_status"] = "unknown"
            elif candidate_years + 0.25 < parsed.minimum_years_experience:
                status["years_experience_status"] = "failed"
            else:
                status["years_experience_status"] = "passed"

        if parsed.degree_requirements:
            degree_rank = self._highest_degree_rank(profile)
            required_rank = max(self._degree_rank(req) for req in parsed.degree_requirements)
            if degree_rank == 0:
                status["degree_status"] = "unknown"
            elif degree_rank < required_rank:
                status["degree_status"] = "failed"
            else:
                status["degree_status"] = "passed"

        if parsed.location:
            location_text = str(profile.get("location_text", ""))
            if not location_text:
                status["location_status"] = "unknown"
            elif parsed.remote_policy.lower() == "remote":
                status["location_status"] = "remote_ok"
            elif parsed.location.lower() in location_text.lower():
                status["location_status"] = "passed"
            else:
                status["location_status"] = "unknown"

        return status

    def _get_candidate_profile(self, candidate_id: str) -> dict:
        parsed = self.parsed_resume_map.get(candidate_id, {})
        meta = self.resume_metadata_by_filename.get(candidate_id, {})
        metadata = parsed.get("metadata", {}) if isinstance(parsed.get("metadata"), dict) else {}
        experience_entries = parsed.get("experience", [])
        education_entries = parsed.get("education", [])
        project_entries = parsed.get("projects", [])
        summary = parsed.get("summary", "")
        text_parts = []
        if meta.get("text"):
            text_parts.append(meta["text"])
        for key in ("summary", "skills", "projects", "certifications"):
            value = parsed.get(key)
            if isinstance(value, list):
                text_parts.append("\n".join(str(item) for item in value))
            elif value:
                text_parts.append(str(value))
        for section_key in ("experience", "education", "projects"):
            for item in parsed.get(section_key, []):
                if isinstance(item, dict) and item.get("raw_text"):
                    text_parts.append(item["raw_text"])

        full_name = metadata.get("full_name", meta.get("full_name", candidate_id))
        major = metadata.get("major", meta.get("major", ""))
        graduation_year = str(metadata.get("graduation_year", meta.get("graduation_year", "")))
        profile = {
            "candidate_id": candidate_id,
            "file_path": meta.get("file_path", parsed.get("file_path", "")),
            "source": meta.get("source", parsed.get("source", "ds3_members")),
            "full_name": full_name,
            "major": major,
            "graduation_year": graduation_year,
            "resume_link": metadata.get("resume_link", meta.get("resume_link", "")),
            "linkedin": metadata.get("linkedin", meta.get("linkedin", "")),
            "github": metadata.get("github", meta.get("github", "")),
            "skills": parsed.get("skills", []),
            "summary": "\n".join(summary) if isinstance(summary, list) else str(summary or ""),
            "education": education_entries,
            "education_entries": education_entries,
            "experience_entries": experience_entries,
            "project_entries": project_entries,
            "employer_names": _extract_employer_names(experience_entries),
            "location_text": (meta.get("text", "")[:300] if meta.get("text") else ""),
            "combined_text": "\n".join(part for part in text_parts if part),
            "estimated_years_experience": self._estimate_years_experience(experience_entries),
        }
        return profile

    def _estimate_years_experience(self, experience_entries: list[dict]) -> float | None:
        if not experience_entries:
            return None
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        spans: list[tuple[int, int]] = []
        current_index = 2026 * 12 + 3
        for entry in experience_entries:
            dates = entry.get("dates", []) if isinstance(entry, dict) else []
            pairs = []
            for date_text in dates:
                parts = re.findall(r"([A-Za-z]{3,9})\.?\s+(\d{4})", str(date_text))
                for month_name, year_text in parts:
                    month = month_map.get(month_name[:3].lower())
                    if month:
                        pairs.append(int(year_text) * 12 + month)
            if len(pairs) >= 2:
                pairs.sort()
                spans.append((pairs[0], pairs[-1]))
            elif len(pairs) == 1:
                spans.append((pairs[0], current_index))
        if not spans:
            return None
        spans.sort()
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        total_months = sum(max(0, end - start) for start, end in merged)
        return round(total_months / 12.0, 1) if total_months > 0 else None

    def _highest_degree_rank(self, profile: dict) -> int:
        education_text = "\n".join(
            item.get("raw_text", "") for item in profile.get("education", []) if isinstance(item, dict)
        )
        combined = f"{profile.get('major', '')}\n{education_text}".lower()
        if re.search(r"\b(phd|doctorate)\b", combined):
            return 3
        if re.search(r"\b(master|m\.s|m\.a)\b", combined):
            return 2
        if re.search(r"\b(bachelor|b\.s|b\.a)\b", combined):
            return 1
        return 0

    def _degree_rank(self, requirement: str) -> int:
        req = requirement.lower()
        if "phd" in req:
            return 3
        if "master" in req:
            return 2
        if "bachelor" in req:
            return 1
        return 0

    def _prepare_chunk_text_index(self):
        self.chunk_term_frequencies = []
        self.chunk_doc_lengths = []
        self.chunk_idf = {}
        self.chunk_avg_doc_length = 0.0
        if not self.chunk_candidates:
            return
        self.chunk_term_frequencies, self.chunk_doc_lengths, self.chunk_idf, self.chunk_avg_doc_length = self._build_text_index(self.chunk_candidates)

    def _prepare_resume_text_index(self):
        self.resume_term_frequencies = []
        self.resume_doc_lengths = []
        self.resume_idf = {}
        self.resume_avg_doc_length = 0.0
        if not self.resume_metadata:
            return
        self.resume_term_frequencies, self.resume_doc_lengths, self.resume_idf, self.resume_avg_doc_length = self._build_text_index(self.resume_metadata)

    def _build_text_index(self, rows: list[dict]):
        term_frequencies: list[dict[str, int]] = []
        doc_lengths: list[int] = []
        document_frequency: dict[str, int] = {}
        for row in rows:
            tokens = _tokenize_text(row.get("text", ""))
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            term_frequencies.append(counts)
            doc_lengths.append(len(tokens))
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        num_docs = len(rows)
        avg_doc_length = sum(doc_lengths) / max(num_docs, 1)
        idf = {
            token: math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
            for token, df in document_frequency.items()
        }
        return term_frequencies, doc_lengths, idf, avg_doc_length

    def _score_bm25(
        self,
        docs: list[dict],
        term_frequencies: list[dict[str, int]],
        doc_lengths: list[int],
        avg_doc_length: float,
        idf_map: dict[str, float],
        query: str,
        query_skills: list[str],
    ) -> np.ndarray:
        if not docs:
            return np.array([], dtype="float32")
        query_tokens = _tokenize_text(query)
        for skill in query_skills:
            query_tokens.extend(_tokenize_text(skill))
        query_terms = list(dict.fromkeys(query_tokens))
        if not query_terms:
            return np.zeros(len(docs), dtype="float32")

        scores = np.zeros(len(docs), dtype="float32")
        k1 = 1.5
        b = 0.75
        for i, (row, tf, doc_length) in enumerate(zip(docs, term_frequencies, doc_lengths)):
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = idf_map.get(term, 0.0)
                denom = freq + k1 * (1 - b + b * doc_length / max(avg_doc_length, 1.0))
                score += idf * ((freq * (k1 + 1)) / max(denom, 1e-6))
            lower_text = row.get("text", "").lower()
            for skill in query_skills:
                if skill.lower() in lower_text:
                    score += 0.75
            scores[i] = score
        max_score = float(scores.max()) if len(scores) else 0.0
        if max_score > 0:
            scores = scores / max_score
        return scores

    def _resume_scores(self, query: str, query_skills: list[str]) -> np.ndarray:
        if self.retrieval_backend == "semantic-resume" and self.index is not None and self.model is not None and self.resume_metadata:
            import faiss as _faiss
            query_embedding = self.model.encode([query], normalize_embeddings=True).astype("float32")
            _faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, len(self.resume_metadata))
            dense = np.zeros(len(self.resume_metadata), dtype="float32")
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    dense[idx] = max(0.0, float(score))
            return dense
        return self._score_bm25(
            docs=self.resume_metadata,
            term_frequencies=self.resume_term_frequencies,
            doc_lengths=self.resume_doc_lengths,
            avg_doc_length=self.resume_avg_doc_length,
            idf_map=self.resume_idf,
            query=query,
            query_skills=query_skills,
        )

    def _search_demo(
        self,
        query: str,
        top_k: int,
        skill_filters: list[str] | None,
        grad_year_filter: str | None,
        major_filter: str | None,
    ) -> list[ResumeResult]:
        query_skills = [s.strip() for s in query.split(",") if s.strip()]
        if skill_filters:
            query_skills = list(dict.fromkeys(query_skills + skill_filters))
        query_tokens = set(_tokenize_text(query))
        results: list[ResumeResult] = []
        for meta in self.resume_metadata:
            text = meta.get("text", "")
            overlap = len(query_tokens & set(_tokenize_text(text)))
            score = overlap / max(len(query_tokens), 1) if query_tokens else 0.0
            if grad_year_filter and str(meta.get("graduation_year", "")) != grad_year_filter:
                continue
            if major_filter and major_filter.lower() not in str(meta.get("major", "")).lower():
                continue
            matched = extract_matched_skills(text, query_skills)
            results.append(
                ResumeResult(
                    rank=0,
                    filename=meta.get("filename", ""),
                    candidate_id=meta.get("filename", ""),
                    score=score,
                    semantic_score=score,
                    file_path=meta.get("file_path", ""),
                    local_resume_path="",
                    text_preview=text[:400],
                    full_text=text,
                    source="demo",
                    full_name=meta.get("full_name", meta.get("filename", "")),
                    major=meta.get("major", ""),
                    graduation_year=meta.get("graduation_year", ""),
                    matched_skills=matched,
                    retrieval_score=score,
                    ranking_details={"mode": "demo_search", "base_search_score": round(score, 4)},
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        for i, item in enumerate(results[:top_k], 1):
            item.rank = i
        return results[:top_k]

    def _build_member_index(self) -> None:
        self._member_index = {}
        if self.members_df is None:
            return
        for _, row in self.members_df.iterrows():
            name = str(row.get("Full Name", "")).strip().lower()
            if name:
                self._member_index[name] = self._row_to_meta(row)

    def _lookup_member(self, filename: str, text: str = "") -> dict:
        if not self._member_index:
            return {}
        name_stem = Path(filename).stem.replace("_", " ").replace("-", " ").lower()
        for name, meta in self._member_index.items():
            if name in name_stem:
                return meta
        if text:
            text_head = text[:500].lower()
            for name, meta in self._member_index.items():
                if name in text_head:
                    return meta
        return {}

    @lru_cache(maxsize=512)
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
        return len(self.resume_metadata)

    def get_unique_majors(self) -> list[str]:
        if self.members_df is not None:
            return sorted(self.members_df["Major"].dropna().unique().tolist())
        return sorted({m.get("major", "") for m in self.resume_metadata if m.get("major")})

    def get_unique_grad_years(self) -> list[str]:
        if self.members_df is not None:
            return sorted(self.members_df["Graduation Year"].dropna().astype(str).unique().tolist())
        return sorted({str(m.get("graduation_year", "")) for m in self.resume_metadata if m.get("graduation_year")})
