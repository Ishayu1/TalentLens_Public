from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.config import MODEL_NAME, PROJECT_ROOT as CONFIG_PROJECT_ROOT
from streamlit.job_description import SKILL_ALIASES

PROJECT_ROOT = CONFIG_PROJECT_ROOT
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTRACTED_PATH = PROCESSED_DIR / "resumes_extracted.json"
PARSED_PATH = PROCESSED_DIR / "resumes_parsed.json"
CHUNKS_PATH = PROCESSED_DIR / "resume_chunks.json"
DS3_CHUNK_EMB_PATH = PROCESSED_DIR / "ds3_chunk_embeddings.npy"
MEMBER_CHUNKS_PATH = PROJECT_ROOT / "member_chunks_metadata.json"
MEMBER_RESUMES_META_PATH = PROJECT_ROOT / "member_resumes_metadata.json"
FAISS_INDEX_PATH = PROJECT_ROOT / "resume_index.faiss"
MEMBER_SOURCES = {"ds3_members", "ds3_board"}
DEFAULT_MEMBER_SOURCE = "ds3_members"
PARSER_VERSION = "ds3_parser_v2"
DEFAULT_GROK_MODE = "auto"
DEFAULT_GROK_WORKERS = 4
ENTRY_REPAIR_CONFIDENCE_THRESHOLD = 0.75
GROK_NON_REASONING_MODELS = [
    "grok-4-fast-non-reasoning",
    "grok-4-1-fast-non-reasoning",
]
SECTION_ALIASES = OrderedDict(
    [
        ("education", ("education", "academic background", "academics", "relevant coursework")),
        ("experience", ("experience", "work experience", "professional experience", "employment", "internship experience")),
        ("projects", ("projects", "project experience", "selected projects", "academic projects", "personal projects")),
        ("skills", ("skills", "technical skills", "skills & interests", "technical skills & interests", "programming languages")),
        ("summary", ("summary", "professional summary", "profile")),
        ("certifications", ("certifications", "licenses", "certificates")),
        ("awards", ("awards", "honors", "honours")),
        ("publications", ("publications", "papers")),
        ("volunteer", ("volunteer", "leadership", "activities", "extracurriculars")),
        ("languages", ("languages",)),
        ("interests", ("interests",)),
        ("contact", ("contact",)),
    ]
)
SECTION_PRIORITY = {
    "contact": 0,
    "summary": 1,
    "education": 2,
    "experience": 3,
    "projects": 4,
    "skills": 5,
    "certifications": 6,
    "awards": 7,
    "publications": 8,
    "volunteer": 9,
    "languages": 10,
    "interests": 11,
}
MONTH_RE = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
DATE_RANGE_RE = re.compile(
    rf"(?i)\b(?:{MONTH_RE}\s+\d{{4}}|\d{{4}})\s*(?:-|–|—|to)\s*(?:{MONTH_RE}\s+\d{{4}}|\d{{4}}|Present|Current)\b"
)
DATE_FRAGMENT_RE = re.compile(rf"(?i)\b(?:{MONTH_RE}\s+\d{{4}}|\d{{4}}|Present|Current)\b")
EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
PHONE_RE = re.compile(r"(?:\+?1[-.\s]*)?(?:\(\d{3}\)|\d{3})[-.\s]*\d{3}[-.\s]*\d{4}")
LOCATION_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1},\s*[A-Z]{2}\b|Remote|Hybrid|On[- ]site", re.I)
LOCATION_SUFFIX_RE = re.compile(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1},\s*[A-Z]{2}|Remote|Hybrid|On[- ]site)$", re.I)
URL_RE = re.compile(r"https?://\S+|linkedin\.com/\S+|github\.com/\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*+•·▪◦]|[A-Za-z]\)|\d+\.)\s*")
TITLE_HINT_RE = re.compile(
    r"(?i)\b(intern|engineer|developer|analyst|assistant|researcher|lead|manager|president|founder|tutor|designer|consultant|scientist|chair|coordinator|officer)\b"
)
TITLE_LOCATION_PREFIXES = {
    "intern",
    "engineer",
    "engineering",
    "developer",
    "analyst",
    "assistant",
    "researcher",
    "manager",
    "president",
    "founder",
    "designer",
    "consultant",
    "scientist",
    "coordinator",
    "officer",
}
ACTION_VERB_RE = re.compile(
    r"(?i)^(?:built|developed|engineered|designed|implemented|led|created|launched|deployed|improved|optimized|"
    r"analyzed|conducted|supported|trained|managed|delivered|reduced|increased|automated|contributed|researched|"
    r"preprocessed|directed|quantified|spearheaded|owned|drove|authored|wrote|performed|generated|presented|integrated)\b"
)
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
DIRTY_SKILL_RE = re.compile(r":|/|\b(?:node js|next_js|sckit|scikit learn|frameworks|libraries)\b", re.I)
SECTION_STOPWORDS = {"skills", "interests", "experience", "projects", "education", "summary", "contact"}
ROLE_PHRASE_HINTS = {
    "software",
    "engineering",
    "engineer",
    "developer",
    "research",
    "science",
    "scientist",
    "machine",
    "learning",
    "analyst",
    "intern",
    "assistant",
    "manager",
    "product",
    "frontend",
    "backend",
    "fullstack",
    "platform",
    "vision",
}
EXTRA_TECH_ALIASES = {
    "Node.js": ("node js", "node.js", "nodejs"),
    "Next.js": ("next js", "next.js", "nextjs", "next_js"),
    "Express.js": ("express js", "express.js"),
    "Tailwind CSS": ("tailwind", "tailwindcss", "tailwind css"),
    "Scikit-learn": ("scikit", "scikit learn", "scikit-learn", "sckit learn", "sklearn"),
    "HuggingFace": ("huggingface", "hugging face"),
    "Pandas": ("pandas",),
    "NumPy": ("numpy",),
    "PyTorch": ("pytorch", "torch"),
    "TensorFlow": ("tensorflow",),
    "Matplotlib": ("matplotlib",),
    "Seaborn": ("seaborn",),
    "FastAPI": ("fastapi",),
    "Flask": ("flask",),
    "Django": ("django",),
    "React": ("react", "react.js", "reactjs"),
    "React Native": ("react native",),
    "TypeScript": ("typescript",),
    "JavaScript": ("javascript",),
    "HTML/CSS": ("html", "css", "html/css"),
    "GitHub Actions": ("github actions",),
    "Git": ("git", "github"),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure",),
    "Google Cloud": ("gcp", "google cloud"),
    "Firebase": ("firebase", "gcp firebase"),
    "Supabase": ("supabase",),
    "MongoDB": ("mongodb", "mongo"),
    "PostgreSQL": ("postgresql", "postgres", "pgvector"),
    "MySQL": ("mysql",),
    "Redis": ("redis",),
    "OpenSearch": ("opensearch",),
    "LangChain": ("langchain",),
    "LangGraph": ("langgraph",),
    "CrewAI": ("crewai",),
    "RAG": ("rag", "retrieval augmented generation"),
    "YOLO": ("yolov8", "yolo"),
    "OAuth2/JWT": ("oauth2/jwt", "oauth", "jwt"),
    "REST APIs": ("rest api", "restful api", "rest apis"),
    "Vercel": ("vercel",),
    "Streamlit": ("streamlit",),
    "Jupyter Notebook": ("jupyter notebook", "jupyter"),
}
SUMMARY_FLAGS_KEY = "summary_flags"
_TECH_PATTERNS: dict[str, list[re.Pattern]] = {}


@dataclass
class RebuildStats:
    total_ds3: int = 0
    entry_grok_candidates: int = 0
    entry_grok_repaired: int = 0
    grok_candidates: int = 0
    grok_enriched: int = 0


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


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


def _clean_resume_text(text: str) -> str:
    text = CONTROL_RE.sub(" ", text or "")
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "-",
        "\u25cf": "-",
        "\u2219": "-",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            lines.append("")
            continue
        line = BULLET_PREFIX_RE.sub("- ", line)
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalize_heading(line: str) -> str:
    normalized = re.sub(r"[^a-z& ]+", " ", line.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _match_section_heading(line: str) -> str | None:
    normalized = _normalize_heading(line)
    if not normalized:
        return None
    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_heading(alias)
            if normalized == alias_norm or normalized.startswith(alias_norm):
                return section
    return None


def _split_sections(text: str) -> dict[str, str]:
    lines = [line.rstrip() for line in _clean_resume_text(text).splitlines()]
    sections: OrderedDict[str, list[str]] = OrderedDict()
    current = "contact"
    sections[current] = []
    for line in lines:
        heading = _match_section_heading(line)
        if heading:
            current = heading
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {
        key: "\n".join(line for line in values if line).strip()
        for key, values in sections.items()
        if any(value.strip() for value in values)
    }


def _build_tech_patterns() -> None:
    if _TECH_PATTERNS:
        return
    alias_map: dict[str, set[str]] = {}
    for canonical, aliases in SKILL_ALIASES.items():
        alias_map.setdefault(canonical, set()).update(str(alias).lower() for alias in aliases)
        alias_map[canonical].add(canonical.lower())
    for canonical, aliases in EXTRA_TECH_ALIASES.items():
        alias_map.setdefault(canonical, set()).update(str(alias).lower() for alias in aliases)
        alias_map[canonical].add(canonical.lower())
    for canonical, aliases in alias_map.items():
        patterns: list[re.Pattern] = []
        for alias in sorted(aliases):
            alias = alias.strip()
            if not alias:
                continue
            escaped = re.escape(alias)
            if re.fullmatch(r"[a-z0-9.+#/ -]+", alias):
                pattern = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I)
            else:
                pattern = re.compile(escaped, re.I)
            patterns.append(pattern)
        _TECH_PATTERNS[canonical] = patterns


def _extract_technologies(text: str) -> list[str]:
    _build_tech_patterns()
    haystack = (text or "").strip()
    if not haystack:
        return []
    matches: list[str] = []
    for canonical, patterns in _TECH_PATTERNS.items():
        if any(pattern.search(haystack) for pattern in patterns):
            matches.append(canonical)
    return sorted(set(matches))


def _normalize_skill_tokens(raw_skills_text: str, raw_tokens: list[str]) -> tuple[list[str], list[str]]:
    canonical = _extract_technologies(raw_skills_text)
    dirty_tokens = [
        token.strip()
        for token in raw_tokens
        if token.strip() and DIRTY_SKILL_RE.search(token.strip())
    ]
    return canonical, dirty_tokens


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _split_skill_tokens(skills_text: str) -> list[str]:
    if not skills_text:
        return []
    normalized = skills_text.replace("\n", ", ")
    normalized = re.sub(r"(?i)\b(?:languages?|frameworks? ?& ?libraries?|frameworks?/libraries?|libraries?|developer tools|business tools|cloud(?: & databases)?|tools(?: & platforms)?|ai/ml|backend|databases|web & frameworks|frameworks & libraries|skills(?: & interests)?|technical skills)\s*:\s*", "", normalized)
    tokens = re.split(r"[,\|;/]+", normalized)
    cleaned: list[str] = []
    for token in tokens:
        value = re.sub(r"\s+", " ", token).strip(" -")
        if value and value.lower() not in SECTION_STOPWORDS:
            cleaned.append(value)
    return cleaned


def _line_has_date(line: str) -> bool:
    return bool(DATE_RANGE_RE.search(line) or len(DATE_FRAGMENT_RE.findall(line)) >= 2)


def _is_bullet_line(line: str) -> bool:
    line = (line or "").strip()
    if not line:
        return False
    if line.startswith("- "):
        return True
    return bool(ACTION_VERB_RE.match(line)) and not _line_has_date(line)


def _merge_bullet_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if not line:
            continue
        if _is_bullet_line(line):
            merged.append(BULLET_PREFIX_RE.sub("", line).strip())
        elif merged:
            merged[-1] = f"{merged[-1]} {line.strip()}".strip()
        else:
            merged.append(line.strip())
    return [re.sub(r"\s+", " ", bullet).strip(" -") for bullet in merged if bullet.strip()]


def _likely_experience_start(lines: list[str], index: int) -> bool:
    line = lines[index].strip()
    next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
    if not line or line.startswith("- ") or _is_bullet_line(line):
        return False
    return _line_has_date(line) or (_line_has_date(next_line) and len(line.split()) <= 12)


def _split_blocks(lines: list[str], start_detector) -> list[list[str]]:
    normalized_lines = [line.strip() for line in lines if line.strip()]
    if not normalized_lines:
        return []
    starts = [idx for idx in range(len(normalized_lines)) if start_detector(normalized_lines, idx)]
    if not starts:
        return [normalized_lines]
    if starts[0] != 0:
        starts = [0] + starts
    blocks: list[list[str]] = []
    for offset, start_idx in enumerate(starts):
        end_idx = starts[offset + 1] if offset + 1 < len(starts) else len(normalized_lines)
        block = normalized_lines[start_idx:end_idx]
        if block:
            blocks.append(block)
    return blocks


def _remove_dates(text: str) -> str:
    cleaned = DATE_RANGE_RE.sub("", text)
    cleaned = DATE_FRAGMENT_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -,|")
    return cleaned


def _extract_location(lines: list[str]) -> str:
    for line in lines:
        match = LOCATION_SUFFIX_RE.search(line.strip())
        if match:
            candidate = match.group(0).strip()
            normalized = _normalize_location_candidate(candidate)
            if normalized:
                return normalized
    return ""


def _normalize_location_candidate(location: str) -> str:
    if not location:
        return ""
    value = location.strip(" -,|")
    if not value:
        return ""
    if value.lower() in {"remote", "hybrid", "on-site", "onsite"}:
        return value
    if "," not in value:
        return value

    city_part, state_part = [part.strip() for part in value.rsplit(",", 1)]
    city_words = city_part.split()
    if len(city_words) >= 2 and city_words[0].lower() in TITLE_LOCATION_PREFIXES:
        city_part = " ".join(city_words[1:]).strip()
    return f"{city_part}, {state_part}".strip(" ,")


def _is_title_like(text: str) -> bool:
    return bool(TITLE_HINT_RE.search(text))


def _parse_header_fields(header_lines: list[str]) -> tuple[str, str, list[str], str]:
    dates: list[str] = []
    for line in header_lines:
        dates.extend(match.group(0) for match in DATE_FRAGMENT_RE.finditer(line))
    cleaned_headers = [_remove_dates(line) for line in header_lines if _remove_dates(line)]
    location = _extract_location(cleaned_headers)
    stripped_headers: list[str] = []
    for line in cleaned_headers:
        stripped_line = line
        if location and stripped_line.endswith(location):
            stripped_line = stripped_line[: -len(location)]
        else:
            stripped_line = LOCATION_SUFFIX_RE.sub("", stripped_line)
        stripped_headers.append(stripped_line.strip(" -,|"))
    header_parts: list[str] = []
    for line in stripped_headers:
        if "|" in line or "//" in line:
            raw_parts = re.split(r"\||//", line)
            header_parts.extend(part.strip(" -,|") for part in raw_parts if part.strip(" -,|"))
        else:
            header_parts.append(line.strip(" -,|"))
    header_parts = [part for part in header_parts if part and not LOCATION_SUFFIX_RE.fullmatch(part)]
    title = ""
    company = ""
    if len(header_parts) >= 2:
        first, second = header_parts[0], header_parts[1]
        if _is_title_like(first) and not _is_title_like(second):
            title, company = first, second
        elif _is_title_like(second) and not _is_title_like(first):
            title, company = second, first
        else:
            title, company = first, second
    elif stripped_headers:
        value = stripped_headers[0]
        if _is_title_like(value):
            title = value
        else:
            company = value
    return title, company, dates, location


def _parse_experience_entries(section_text: str) -> list[dict]:
    if not section_text:
        return []
    lines = [line.strip() for line in _clean_resume_text(section_text).splitlines() if line.strip()]
    blocks = _split_blocks(lines, _likely_experience_start)
    entries: list[dict] = []
    for block in blocks:
        header_lines: list[str] = []
        body_lines: list[str] = []
        for line in block:
            if not body_lines and not _is_bullet_line(line) and len(header_lines) < 2:
                header_lines.append(line)
                continue
            body_lines.append(line)
        if not body_lines and header_lines:
            body_lines = header_lines[1:]
            header_lines = header_lines[:1]
        bullets = _merge_bullet_lines(body_lines)
        title, company, dates, location = _parse_header_fields(header_lines)
        raw_text = "\n".join(block).strip()
        technologies = _extract_technologies("\n".join(header_lines + bullets))
        normalized_company = _normalize_company_name(company)
        if normalized_company:
            technologies = [
                tech for tech in technologies
                if _normalize_company_name(tech) != normalized_company
            ]
        entries.append(
            _annotate_entry_metadata(
                {
                    "title": title,
                    "company": company,
                    "company_normalized": normalized_company,
                    "dates": dates,
                    "location": location,
                    "bullets": bullets,
                    "technologies": technologies,
                    "raw_header": " | ".join(header_lines).strip(),
                    "raw_text": raw_text,
                },
                "experience",
            )
        )
    return entries


def _likely_project_start(lines: list[str], index: int) -> bool:
    line = lines[index].strip()
    if not line or line.startswith("- "):
        return False
    if "|" in line:
        return True
    if _line_has_date(line):
        return True
    if index + 1 < len(lines) and _is_bullet_line(lines[index + 1]):
        return len(line.split()) <= 16
    return False


def _parse_project_entries(section_text: str) -> list[dict]:
    if not section_text:
        return []
    lines = [line.strip() for line in _clean_resume_text(section_text).splitlines() if line.strip()]
    blocks = _split_blocks(lines, _likely_project_start)
    entries: list[dict] = []
    for block in blocks:
        header_lines: list[str] = []
        body_lines: list[str] = []
        for line in block:
            if not body_lines and not _is_bullet_line(line) and len(header_lines) < 2:
                header_lines.append(line)
                continue
            body_lines.append(line)
        if not body_lines and header_lines:
            body_lines = header_lines[1:]
            header_lines = header_lines[:1]
        header = header_lines[0] if header_lines else ""
        dates = [match.group(0) for match in DATE_FRAGMENT_RE.finditer("\n".join(header_lines))]
        location = _extract_location(header_lines)
        header_no_dates = _remove_dates(LOCATION_RE.sub("", header))
        name = header_no_dates.split("|")[0].strip(" -,")
        bullets = _merge_bullet_lines(body_lines)
        technologies = _extract_technologies("\n".join(header_lines + bullets))
        entries.append(
            _annotate_entry_metadata(
                {
                    "name": name,
                    "dates": dates,
                    "location": location,
                    "bullets": bullets,
                    "technologies": technologies,
                    "raw_header": " | ".join(header_lines).strip(),
                    "raw_text": "\n".join(block).strip(),
                },
                "projects",
            )
        )
    return entries


def _entry_header_lines(entry: dict) -> list[str]:
    raw_text = str(entry.get("raw_text", "")).strip()
    header_lines: list[str] = []
    if raw_text:
        for line in _clean_resume_text(raw_text).splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _is_bullet_line(stripped):
                break
            header_lines.append(stripped)
            if len(header_lines) >= 2:
                break
    if not header_lines:
        raw_header = str(entry.get("raw_header", "")).strip()
        if raw_header:
            header_lines = [part.strip() for part in raw_header.split("|") if part.strip()][:2]
    return header_lines


def _looks_like_role_phrase(text: str) -> bool:
    normalized = _normalize_company_name(text)
    if not normalized:
        return False
    if _is_title_like(text):
        return True
    return any(token in ROLE_PHRASE_HINTS for token in normalized.split())


def _is_generic_company_label(text: str) -> bool:
    normalized = _normalize_company_name(text)
    if not normalized:
        return False
    tokens = normalized.split()
    if len(tokens) > 5:
        return False
    if normalized in {"software engineering", "data science", "machine learning", "computer vision"}:
        return True
    return _looks_like_role_phrase(text) and not any(
        token in {"lab", "institute", "group", "technologies", "systems", "society", "university"}
        for token in tokens
    )


def _score_from_warnings(base: float, warnings: list[str]) -> float:
    penalties = {
        "missing_title": 0.25,
        "missing_company": 0.28,
        "missing_company_normalized": 0.18,
        "missing_project_name": 0.28,
        "company_looks_like_title": 0.35,
        "title_company_inversion": 0.35,
        "company_then_title_layout": 0.20,
        "sparse_structured_parse": 0.18,
        "title_like_location": 0.30,
        "project_name_too_generic": 0.20,
        "project_header_sparse": 0.18,
    }
    score = base
    for warning in warnings:
        score -= penalties.get(warning, 0.08)
    return round(max(0.05, min(0.99, score)), 2)


def _experience_entry_warnings(entry: dict) -> list[str]:
    warnings: list[str] = []
    title = str(entry.get("title", "")).strip()
    company = str(entry.get("company", "")).strip()
    company_normalized = str(entry.get("company_normalized", "")).strip()
    location = str(entry.get("location", "")).strip()
    header_lines = _entry_header_lines(entry)
    cleaned_header_lines = [_remove_dates(line) for line in header_lines if _remove_dates(line)]

    if not title:
        warnings.append("missing_title")
    if not company:
        warnings.append("missing_company")
    elif _is_generic_company_label(company):
        warnings.append("company_looks_like_title")
    if company and title and not _looks_like_role_phrase(title) and _looks_like_role_phrase(company):
        warnings.append("title_company_inversion")
    if not company_normalized:
        warnings.append("missing_company_normalized")
    if location:
        first_word = location.split()[0].strip(",").lower()
        if first_word in TITLE_LOCATION_PREFIXES:
            warnings.append("title_like_location")
    if len(cleaned_header_lines) >= 2:
        first, second = cleaned_header_lines[0], cleaned_header_lines[1]
        if first and second and not _looks_like_role_phrase(first) and _looks_like_role_phrase(second):
            warnings.append("company_then_title_layout")
    raw_header = str(entry.get("raw_header", "")).strip()
    if raw_header and ("|" in raw_header or len(header_lines) >= 2) and (not title or not company):
        warnings.append("sparse_structured_parse")
    return sorted(set(warnings))


def _project_entry_warnings(entry: dict) -> list[str]:
    warnings: list[str] = []
    name = str(entry.get("name", "")).strip()
    raw_header = str(entry.get("raw_header", "")).strip()
    location = str(entry.get("location", "")).strip()

    if not name:
        warnings.append("missing_project_name")
    elif len(name.split()) <= 2 and name.lower() in {"project", "projects", "research", "application"}:
        warnings.append("project_name_too_generic")
    if raw_header and ("|" in raw_header or _line_has_date(raw_header)) and not name:
        warnings.append("project_header_sparse")
    if location:
        first_word = location.split()[0].strip(",").lower()
        if first_word in TITLE_LOCATION_PREFIXES:
            warnings.append("title_like_location")
    return sorted(set(warnings))


def _annotate_entry_metadata(
    entry: dict,
    section_type: str,
    repair_source: str = "deterministic",
    suppress_warnings: set[str] | None = None,
) -> dict:
    annotated = deepcopy(entry)
    if section_type == "experience":
        warnings = _experience_entry_warnings(annotated)
        base_confidence = 0.9 if repair_source == "deterministic" else 0.95
    else:
        warnings = _project_entry_warnings(annotated)
        base_confidence = 0.92 if repair_source == "deterministic" else 0.96
    if suppress_warnings:
        warnings = [warning for warning in warnings if warning not in suppress_warnings]
    annotated["entry_parse_warnings"] = warnings
    annotated["entry_parse_confidence"] = _score_from_warnings(base_confidence, warnings)
    annotated["repair_source"] = repair_source
    return annotated


def _entry_needs_grok_repair(entry: dict) -> bool:
    return bool(entry.get("entry_parse_warnings")) and float(entry.get("entry_parse_confidence", 0.0)) < ENTRY_REPAIR_CONFIDENCE_THRESHOLD


def _parse_education_entries(section_text: str) -> list[dict]:
    if not section_text:
        return []
    lines = [line.strip() for line in _clean_resume_text(section_text).splitlines() if line.strip()]
    blocks = _split_blocks(lines, lambda rows, idx: _line_has_date(rows[idx]) or (idx + 1 < len(rows) and _line_has_date(rows[idx + 1])))
    entries: list[dict] = []
    for block in blocks:
        header = block[:3]
        bullets = _merge_bullet_lines(block[3:])
        dates = [match.group(0) for match in DATE_FRAGMENT_RE.finditer("\n".join(header))]
        entries.append(
            {
                "raw_header": " | ".join(header).strip(),
                "dates": dates,
                "bullets": bullets,
                "raw_text": "\n".join(block).strip(),
            }
        )
    return entries


def _extract_contact(text: str) -> str:
    lines = [line.strip() for line in _clean_resume_text(text).splitlines() if line.strip()]
    contact_lines: list[str] = []
    for line in lines[:6]:
        if EMAIL_RE.search(line) or PHONE_RE.search(line) or URL_RE.search(line):
            contact_lines.append(line)
        elif not contact_lines:
            contact_lines.append(line)
    return "\n".join(contact_lines[:3]).strip()


def _compute_parse_confidence(parsed: dict) -> float:
    score = 0.2
    if parsed.get("canonical_skills"):
        score += 0.25
    if len(parsed.get("experience", [])) >= 2:
        score += 0.25
    if parsed.get("projects"):
        score += 0.15
    if all(entry.get("company_normalized") for entry in parsed.get("experience", [])):
        score += 0.15
    entry_confidences = [
        float(entry.get("entry_parse_confidence", 0.0))
        for entry in parsed.get("experience", []) + parsed.get("projects", [])
        if isinstance(entry, dict)
    ]
    if entry_confidences:
        score += 0.10 * min(1.0, sum(entry_confidences) / len(entry_confidences))
    return round(min(score, 0.95), 2)


def _should_enrich_with_grok(parsed: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    skills = parsed.get("canonical_skills", [])
    experience = parsed.get("experience", [])
    projects = parsed.get("projects", [])
    sections_raw = parsed.get("sections_raw", {})
    if len(skills) < 5:
        reasons.append("low_skill_count")
    if len(experience) <= 1 and len(sections_raw.get("experience", "")) > 250:
        reasons.append("undersplit_experience")
    if sections_raw.get("projects") and len(projects) <= 1 and len(sections_raw.get("projects", "")) > 180:
        reasons.append("undersplit_projects")
    if any(not entry.get("company_normalized") for entry in experience):
        reasons.append("missing_company")
    if any(_entry_needs_grok_repair(entry) for entry in experience + projects if isinstance(entry, dict)):
        reasons.append("ambiguous_entries")
    if parsed.get("parse_warnings"):
        reasons.append("parse_warnings_present")
    return bool(reasons), reasons


def _grok_api_key() -> str | None:
    return os.getenv("XAI_API_KEY")


def _grok_json_payload(raw_text: str) -> dict:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        lines = cleaned.splitlines()
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in Grok enrichment response")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Grok enrichment response must be a JSON object")
    return payload


def _call_non_reasoning_grok(prompt: str) -> dict:
    api_key = _grok_api_key()
    if not api_key:
        raise RuntimeError("XAI_API_KEY is not configured")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    session = requests.Session()
    for model_name in GROK_NON_REASONING_MODELS:
        try:
            response = session.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You extract structured resume data. Return strict JSON only. "
                                "Do not invent missing facts. Prefer empty strings or empty arrays."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            return _grok_json_payload(response.json()["choices"][0]["message"]["content"])
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Grok enrichment failed: {last_error}")


def _grok_entry_prompt(candidate_id: str, section_type: str, entry: dict) -> str:
    header_lines = _entry_header_lines(entry)
    prompt_payload = {
        "candidate_id": candidate_id,
        "section_type": section_type,
        "raw_header_lines": header_lines,
        "raw_header": entry.get("raw_header", ""),
        "raw_text": entry.get("raw_text", "")[:2500],
        "deterministic_parse": entry,
    }
    if section_type == "experience":
        expected_keys = (
            "{\"title\": string, \"company\": string, \"company_normalized\": string, "
            "\"dates\": string[], \"location\": string, \"bullets\": string[], "
            "\"technologies\": string[], \"confidence\": number, \"reason\": string}"
        )
    else:
        expected_keys = (
            "{\"name\": string, \"dates\": string[], \"location\": string, "
            "\"bullets\": string[], \"technologies\": string[], "
            "\"confidence\": number, \"reason\": string}"
        )
    return (
        f"Repair this single {section_type} entry from a DS3 resume. Use only the provided text. "
        "Do not hallucinate missing facts. Prefer empty strings or empty arrays when uncertain. "
        f"Return strict JSON only with this exact shape: {expected_keys}\n\n"
        "Rules:\n"
        "- fix title/company inversions when header layout is ambiguous\n"
        "- keep dates and location grounded in the header text\n"
        "- preserve bullets unless the provided raw text clearly supports a cleaner equivalent\n"
        "- technologies must be canonical names when they are explicitly supported\n\n"
        f"Input:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
    )


def _merge_grok_entry_repair(entry: dict, payload: dict, section_type: str) -> dict:
    merged = deepcopy(entry)
    original_bullets = _coerce_string_list(entry.get("bullets"))
    payload_bullets = _coerce_string_list(payload.get("bullets"))
    bullets = original_bullets or payload_bullets

    if section_type == "experience":
        title = str(payload.get("title", "")).strip() or str(entry.get("title", "")).strip()
        company = str(payload.get("company", "")).strip() or str(entry.get("company", "")).strip()
        merged.update(
            {
                "title": title,
                "company": company,
                "company_normalized": (
                    str(payload.get("company_normalized", "")).strip()
                    or _normalize_company_name(company)
                    or str(entry.get("company_normalized", "")).strip()
                ),
                "dates": _coerce_string_list(payload.get("dates")) or _coerce_string_list(entry.get("dates")),
                "location": str(payload.get("location", "")).strip() or str(entry.get("location", "")).strip(),
                "bullets": bullets,
            }
        )
        technology_text = "\n".join(
            [
                str(merged.get("title", "")).strip(),
                str(merged.get("company", "")).strip(),
                str(merged.get("raw_header", "")).strip(),
                "\n".join(bullets),
                " ".join(_coerce_string_list(payload.get("technologies"))),
            ]
        )
        merged["technologies"] = sorted(
            set(
                _coerce_string_list(entry.get("technologies"))
                + _coerce_string_list(payload.get("technologies"))
                + _extract_technologies(technology_text)
            )
        )
        merged = _annotate_entry_metadata(
            merged,
            "experience",
            repair_source="grok_entry_repair",
            suppress_warnings={"company_then_title_layout", "company_looks_like_title", "title_company_inversion", "sparse_structured_parse"},
        )
    else:
        name = str(payload.get("name", "")).strip() or str(entry.get("name", "")).strip()
        merged.update(
            {
                "name": name,
                "dates": _coerce_string_list(payload.get("dates")) or _coerce_string_list(entry.get("dates")),
                "location": str(payload.get("location", "")).strip() or str(entry.get("location", "")).strip(),
                "bullets": bullets,
            }
        )
        technology_text = "\n".join(
            [
                str(merged.get("name", "")).strip(),
                str(merged.get("raw_header", "")).strip(),
                "\n".join(bullets),
                " ".join(_coerce_string_list(payload.get("technologies"))),
            ]
        )
        merged["technologies"] = sorted(
            set(
                _coerce_string_list(entry.get("technologies"))
                + _coerce_string_list(payload.get("technologies"))
                + _extract_technologies(technology_text)
            )
        )
        merged = _annotate_entry_metadata(
            merged,
            "projects",
            repair_source="grok_entry_repair",
            suppress_warnings={"project_header_sparse", "project_name_too_generic"},
        )

    try:
        payload_confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        payload_confidence = 0.0
    if payload_confidence > 0:
        merged["entry_parse_confidence"] = round(
            max(float(merged.get("entry_parse_confidence", 0.0)), min(payload_confidence, 0.99)),
            2,
        )
    reason = str(payload.get("reason", "")).strip()
    if reason:
        warnings = list(merged.get("entry_parse_warnings", []))
        warnings.append(f"grok_entry_repair_reason:{reason}")
        merged["entry_parse_warnings"] = sorted(set(warnings))
    return merged


def _grok_prompt(extracted_record: dict, parsed: dict, reasons: list[str]) -> str:
    payload = {
        "candidate_id": parsed.get("candidate_id"),
        "reasons": reasons,
        "sections_raw": parsed.get("sections_raw", {}),
        "deterministic_parse": {
            "canonical_skills": parsed.get("canonical_skills", []),
            "experience": parsed.get("experience", []),
            "projects": parsed.get("projects", []),
            "parse_warnings": parsed.get("parse_warnings", []),
        },
        "resume_excerpt": extracted_record.get("text", "")[:7000],
    }
    return (
        "Repair this DS3 student resume parse. Preserve valid deterministic output and fill only what can be grounded "
        "in the resume text. Return strict JSON with these keys only:\n"
        "{"
        "\"canonical_skills\": string[], "
        "\"experience_entries\": [{\"title\": string, \"company\": string, \"company_normalized\": string, \"dates\": string[], \"location\": string, \"bullets\": string[], \"technologies\": string[]}], "
        "\"project_entries\": [{\"name\": string, \"dates\": string[], \"location\": string, \"bullets\": string[], \"technologies\": string[]}], "
        "\"summary_flags\": string[], "
        "\"parse_warnings\": string[]"
        "}\n\n"
        "Rules:\n"
        "- do not hallucinate employers or technologies\n"
        "- split collapsed experience and project blocks into separate entries when the text clearly contains multiple roles/projects\n"
        "- normalize company names but keep company_normalized empty if uncertain\n"
        "- technologies must be canonical names\n"
        "- summary_flags can include unclear_company, unclear_dates, underspecified_projects, low_skill_signal\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _merge_grok_enrichment(parsed: dict, payload: dict) -> dict:
    enriched = deepcopy(parsed)
    canonical_skills = sorted(
        set(
            parsed.get("canonical_skills", [])
            + _extract_technologies(" ".join(_coerce_string_list(payload.get("canonical_skills"))))
            + _coerce_string_list(payload.get("canonical_skills"))
        )
    )
    if canonical_skills:
        enriched["canonical_skills"] = canonical_skills
    experience_entries = payload.get("experience_entries")
    if isinstance(experience_entries, list) and experience_entries:
        enriched["experience"] = [
            _annotate_entry_metadata(
                {
                    "title": str(entry.get("title", "")).strip(),
                    "company": str(entry.get("company", "")).strip(),
                    "company_normalized": str(entry.get("company_normalized", "")).strip() or _normalize_company_name(str(entry.get("company", ""))),
                    "dates": _coerce_string_list(entry.get("dates")),
                    "location": str(entry.get("location", "")).strip(),
                    "bullets": _coerce_string_list(entry.get("bullets")),
                    "technologies": sorted(
                        set(
                            _extract_technologies(" ".join(_coerce_string_list(entry.get("technologies"))))
                            + _coerce_string_list(entry.get("technologies"))
                        )
                    ),
                    "raw_header": str(entry.get("title", "")).strip(),
                    "raw_text": "\n".join(
                        [str(entry.get("title", "")).strip(), str(entry.get("company", "")).strip()]
                        + _coerce_string_list(entry.get("bullets"))
                    ).strip(),
                },
                "experience",
                repair_source="grok_resume_enrichment",
            )
            for entry in experience_entries
            if isinstance(entry, dict)
        ]
    project_entries = payload.get("project_entries")
    if isinstance(project_entries, list) and project_entries:
        enriched["projects"] = [
            _annotate_entry_metadata(
                {
                    "name": str(entry.get("name", "")).strip(),
                    "dates": _coerce_string_list(entry.get("dates")),
                    "location": str(entry.get("location", "")).strip(),
                    "bullets": _coerce_string_list(entry.get("bullets")),
                    "technologies": sorted(
                        set(
                            _extract_technologies(" ".join(_coerce_string_list(entry.get("technologies"))))
                            + _coerce_string_list(entry.get("technologies"))
                        )
                    ),
                    "raw_header": str(entry.get("name", "")).strip(),
                    "raw_text": "\n".join(
                        [str(entry.get("name", "")).strip()] + _coerce_string_list(entry.get("bullets"))
                    ).strip(),
                },
                "projects",
                repair_source="grok_resume_enrichment",
            )
            for entry in project_entries
            if isinstance(entry, dict)
        ]
    warnings = []
    warnings.extend(str(flag).strip() for flag in payload.get(SUMMARY_FLAGS_KEY, []) if str(flag).strip())
    warnings.extend(str(flag).strip() for flag in payload.get("parse_warnings", []) if str(flag).strip())
    enriched["parse_warnings"] = sorted(set(enriched.get("parse_warnings", []) + warnings))
    experience_tech = sorted({tech for entry in enriched.get("experience", []) for tech in entry.get("technologies", [])})
    project_tech = sorted({tech for entry in enriched.get("projects", []) for tech in entry.get("technologies", [])})
    enriched["technology_sources"] = {
        "skills_section": sorted(set(enriched.get("technology_sources", {}).get("skills_section", []))),
        "experience": experience_tech,
        "projects": project_tech,
    }
    enriched["canonical_skills"] = sorted(set(enriched.get("canonical_skills", []) + experience_tech + project_tech))
    enriched["parser_version"] = PARSER_VERSION
    enriched["parse_confidence"] = _compute_parse_confidence(enriched)
    return enriched


def _deterministic_parse(record: dict) -> dict:
    text = _clean_resume_text(record.get("text", ""))
    sections_raw = _split_sections(text)
    summary_text = sections_raw.get("summary", "")
    skills_section = sections_raw.get("skills", "")
    raw_skill_tokens = _split_skill_tokens(skills_section)
    experience_entries = _parse_experience_entries(sections_raw.get("experience", ""))
    project_entries = _parse_project_entries(sections_raw.get("projects", ""))
    education_entries = _parse_education_entries(sections_raw.get("education", ""))
    canonical_skills, dirty_tokens = _normalize_skill_tokens(
        "\n".join(
            [
                skills_section,
                sections_raw.get("experience", ""),
                sections_raw.get("projects", ""),
            ]
        ),
        raw_skill_tokens,
    )
    experience_tech = sorted({tech for entry in experience_entries for tech in entry.get("technologies", [])})
    project_tech = sorted({tech for entry in project_entries for tech in entry.get("technologies", [])})
    canonical_skills = sorted(set(canonical_skills + experience_tech + project_tech))
    warnings: list[str] = []
    if dirty_tokens:
        warnings.append("dirty_skill_tokens")
    if len(experience_entries) <= 1 and len(sections_raw.get("experience", "")) > 250:
        warnings.append("experience_not_split")
    if sections_raw.get("projects") and len(project_entries) <= 1 and len(sections_raw.get("projects", "")) > 180:
        warnings.append("projects_not_split")
    if any(not entry.get("company_normalized") for entry in experience_entries):
        warnings.append("missing_company_normalized")
    metadata = deepcopy(record.get("metadata", {})) if isinstance(record.get("metadata"), dict) else {}
    parsed = {
        "candidate_id": record.get("filename", ""),
        "source": record.get("source", ""),
        "file_path": record.get("file_path", ""),
        "word_count": record.get("word_count", 0),
        "metadata": metadata,
        "contact": _extract_contact(text),
        "summary": summary_text,
        "education": education_entries,
        "experience": experience_entries,
        "projects": project_entries,
        "skills": raw_skill_tokens,
        "canonical_skills": canonical_skills,
        "technology_sources": {
            "skills_section": _extract_technologies(skills_section),
            "experience": experience_tech,
            "projects": project_tech,
        },
        "certifications": _split_skill_tokens(sections_raw.get("certifications", "")),
        "sections_raw": sections_raw,
        "parse_warnings": sorted(set(warnings)),
        "parser_version": PARSER_VERSION,
    }
    parsed["parse_confidence"] = _compute_parse_confidence(parsed)
    return parsed


def _build_entry_text(prefix: str, header: str, company: str, technologies: list[str], bullets: list[str]) -> str:
    lines = [prefix]
    if header:
        lines.append(header)
    if company:
        lines.append(f"Company: {company}")
    if technologies:
        lines.append("Technologies: " + ", ".join(technologies))
    lines.extend(f"- {bullet}" for bullet in bullets if bullet)
    return "\n".join(line for line in lines if line).strip()


def _build_chunks(parsed: dict) -> list[dict]:
    candidate_id = parsed.get("candidate_id", "")
    source = parsed.get("source", DEFAULT_MEMBER_SOURCE)
    chunks: list[dict] = []

    def add_chunk(section_type: str, text: str, metadata: dict[str, Any] | None = None):
        chunk_text = (text or "").strip()
        if not chunk_text:
            return
        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "candidate_id": candidate_id,
                "source": source,
                "section_type": section_type,
                "text": chunk_text,
                "metadata": {
                    "candidate_id": candidate_id,
                    "source": source,
                    **(metadata or {}),
                },
            }
        )

    add_chunk("contact", parsed.get("contact", ""))
    add_chunk("summary", parsed.get("summary", ""))
    for entry in parsed.get("education", []):
        add_chunk("education", entry.get("raw_text", ""), {"dates": entry.get("dates", [])})
    for entry in parsed.get("experience", []):
        add_chunk(
            "experience",
            _build_entry_text(
                prefix=f"Role: {entry.get('title', '')}".strip(),
                header=entry.get("raw_header", ""),
                company=entry.get("company", ""),
                technologies=entry.get("technologies", []),
                bullets=entry.get("bullets", []),
            ),
            {
                "dates": entry.get("dates", []),
                "company": entry.get("company", ""),
                "company_normalized": entry.get("company_normalized", ""),
                "technologies": entry.get("technologies", []),
            },
        )
    for entry in parsed.get("projects", []):
        add_chunk(
            "projects",
            _build_entry_text(
                prefix=f"Project: {entry.get('name', '')}".strip(),
                header=entry.get("raw_header", ""),
                company="",
                technologies=entry.get("technologies", []),
                bullets=entry.get("bullets", []),
            ),
            {
                "dates": entry.get("dates", []),
                "technologies": entry.get("technologies", []),
            },
        )
    add_chunk(
        "skills",
        ", ".join(parsed.get("canonical_skills", [])),
        {
            "skills_list": parsed.get("canonical_skills", []),
            "technologies": parsed.get("canonical_skills", []),
        },
    )
    for section_name in ("certifications", "awards", "publications", "volunteer", "languages", "interests"):
        raw_text = parsed.get("sections_raw", {}).get(section_name, "")
        add_chunk(section_name, raw_text)
    return chunks


def _member_resume_metadata(ds3_records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in ds3_records:
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        rows.append(
            {
                "filename": record.get("filename", ""),
                "file_path": record.get("file_path", ""),
                "text": record.get("text", ""),
                "source": record.get("source", DEFAULT_MEMBER_SOURCE),
                "full_name": metadata.get("full_name", ""),
                "major": metadata.get("major", ""),
                "graduation_year": metadata.get("graduation_year", ""),
                "resume_link": metadata.get("resume_link", ""),
                "linkedin": metadata.get("linkedin", ""),
                "github": metadata.get("github", ""),
            }
        )
    return rows


def _encode_and_index_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "Embedding dependencies are unavailable in this Python environment. "
            "Run the rebuild with the project virtualenv, for example: "
            "./venv/bin/python pipeline/ds3_rebuild.py --use-grok never"
        ) from exc

    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    texts = [chunk.get("text", "") for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    np.save(DS3_CHUNK_EMB_PATH, embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))


def _repair_ambiguous_entries_with_grok(
    extracted_by_id: dict[str, dict],
    parsed_by_id: dict[str, dict],
    use_grok: str,
    grok_workers: int,
) -> tuple[dict[str, dict], RebuildStats]:
    stats = RebuildStats(total_ds3=len(parsed_by_id))
    if use_grok == "never":
        return parsed_by_id, stats

    api_key = _grok_api_key()
    if use_grok == "always" and not api_key:
        raise RuntimeError("XAI_API_KEY is required when --use-grok=always")
    if use_grok == "auto" and not api_key:
        return parsed_by_id, stats

    tasks: list[tuple[str, str, int, dict]] = []
    for candidate_id, parsed in parsed_by_id.items():
        for section_type, entries_key in (("experience", "experience"), ("projects", "projects")):
            entries = parsed.get(entries_key, [])
            for index, entry in enumerate(entries):
                if isinstance(entry, dict) and _entry_needs_grok_repair(entry):
                    stats.entry_grok_candidates += 1
                    tasks.append((candidate_id, section_type, index, entry))

    def run_task(task: tuple[str, str, int, dict]) -> tuple[str, str, int, dict]:
        candidate_id, section_type, index, entry = task
        try:
            payload = _call_non_reasoning_grok(_grok_entry_prompt(candidate_id, section_type, entry))
            repaired = _merge_grok_entry_repair(entry, payload, section_type)
            return candidate_id, section_type, index, repaired
        except Exception as exc:
            fallback = deepcopy(entry)
            warnings = list(fallback.get("entry_parse_warnings", []))
            warnings.append(f"grok_entry_repair_failed:{exc.__class__.__name__}")
            fallback["entry_parse_warnings"] = sorted(set(warnings))
            return candidate_id, section_type, index, fallback

    if not tasks:
        return parsed_by_id, stats

    with ThreadPoolExecutor(max_workers=max(1, grok_workers)) as executor:
        future_map = {executor.submit(run_task, task): task for task in tasks}
        with tqdm(total=len(future_map), desc="Grok entry repair", unit="entry") as progress:
            for future in as_completed(future_map):
                candidate_id, section_type, index, repaired = future.result()
                entries_key = "experience" if section_type == "experience" else "projects"
                parsed_by_id[candidate_id][entries_key][index] = repaired
                if repaired.get("repair_source") == "grok_entry_repair":
                    stats.entry_grok_repaired += 1
                progress.update(1)

    for parsed in parsed_by_id.values():
        parsed["parse_confidence"] = _compute_parse_confidence(parsed)
    return parsed_by_id, stats


def _enrich_ds3_records(
    extracted_by_id: dict[str, dict],
    parsed_by_id: dict[str, dict],
    use_grok: str,
    grok_workers: int,
) -> tuple[dict[str, dict], RebuildStats]:
    stats = RebuildStats(total_ds3=len(parsed_by_id))
    if use_grok == "never":
        return parsed_by_id, stats

    api_key = _grok_api_key()
    if use_grok == "always" and not api_key:
        raise RuntimeError("XAI_API_KEY is required when --use-grok=always")
    if use_grok == "auto" and not api_key:
        return parsed_by_id, stats

    tasks: list[tuple[str, dict, list[str]]] = []
    for candidate_id, parsed in parsed_by_id.items():
        should_enrich, reasons = _should_enrich_with_grok(parsed)
        if should_enrich:
            stats.grok_candidates += 1
            tasks.append((candidate_id, parsed, reasons))

    def run_task(task: tuple[str, dict, list[str]]) -> tuple[str, dict | None]:
        candidate_id, parsed, reasons = task
        try:
            payload = _call_non_reasoning_grok(_grok_prompt(extracted_by_id[candidate_id], parsed, reasons))
            return candidate_id, _merge_grok_enrichment(parsed, payload)
        except Exception as exc:
            fallback = deepcopy(parsed)
            warnings = sorted(set(fallback.get("parse_warnings", []) + [f"grok_enrichment_failed:{exc.__class__.__name__}"]))
            fallback["parse_warnings"] = warnings
            return candidate_id, fallback

    if not tasks:
        return parsed_by_id, stats

    enriched: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, grok_workers)) as executor:
        future_map = {executor.submit(run_task, task): task[0] for task in tasks}
        with tqdm(total=len(future_map), desc="Grok enrichment", unit="resume") as progress:
            for future in as_completed(future_map):
                candidate_id, parsed = future.result()
                if parsed is not None:
                    stats.grok_enriched += 1
                    enriched[candidate_id] = parsed
                progress.update(1)

    for candidate_id, parsed in enriched.items():
        parsed_by_id[candidate_id] = parsed
    return parsed_by_id, stats


def rebuild_ds3_artifacts(use_grok: str = DEFAULT_GROK_MODE, grok_workers: int = DEFAULT_GROK_WORKERS) -> RebuildStats:
    extracted_rows = _load_json(EXTRACTED_PATH)
    existing_parsed = _load_json(PARSED_PATH)
    existing_chunks = _load_json(CHUNKS_PATH)
    ds3_extracted = [row for row in extracted_rows if row.get("source") in MEMBER_SOURCES]
    extracted_by_id = {row.get("filename", ""): row for row in ds3_extracted if row.get("filename")}

    parsed_by_id: dict[str, dict] = {}
    for record_id, record in tqdm(
        extracted_by_id.items(),
        total=len(extracted_by_id),
        desc="Deterministic parsing",
        unit="resume",
    ):
        parsed_by_id[record_id] = _deterministic_parse(record)
    parsed_by_id, entry_stats = _repair_ambiguous_entries_with_grok(
        extracted_by_id,
        parsed_by_id,
        use_grok=use_grok,
        grok_workers=grok_workers,
    )
    parsed_by_id, stats = _enrich_ds3_records(extracted_by_id, parsed_by_id, use_grok=use_grok, grok_workers=grok_workers)
    stats.entry_grok_candidates = entry_stats.entry_grok_candidates
    stats.entry_grok_repaired = entry_stats.entry_grok_repaired

    ds3_parsed_rows = [parsed_by_id[record_id] for record_id in extracted_by_id]
    ds3_chunks: list[dict] = []
    for parsed in tqdm(ds3_parsed_rows, total=len(ds3_parsed_rows), desc="Building chunks", unit="resume"):
        ds3_chunks.extend(_build_chunks(parsed))

    non_ds3_parsed = [row for row in existing_parsed if row.get("source") not in MEMBER_SOURCES]
    merged_parsed = ds3_parsed_rows + non_ds3_parsed
    non_ds3_chunks = [row for row in existing_chunks if row.get("source") not in MEMBER_SOURCES]
    merged_chunks = ds3_chunks + non_ds3_chunks

    _dump_json(PARSED_PATH, merged_parsed)
    _dump_json(CHUNKS_PATH, merged_chunks)
    _dump_json(MEMBER_CHUNKS_PATH, ds3_chunks)
    _dump_json(MEMBER_RESUMES_META_PATH, _member_resume_metadata(ds3_extracted))
    _encode_and_index_chunks(ds3_chunks)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild DS3 parsed resumes, chunks, and FAISS artifacts.")
    parser.add_argument(
        "--use-grok",
        choices=["auto", "always", "never"],
        default=DEFAULT_GROK_MODE,
        help="Whether to apply non-reasoning Grok enrichment to weak DS3 parses.",
    )
    parser.add_argument(
        "--grok-workers",
        type=int,
        default=DEFAULT_GROK_WORKERS,
        help="Parallel Grok worker count for offline enrichment.",
    )
    args = parser.parse_args()

    stats = rebuild_ds3_artifacts(use_grok=args.use_grok, grok_workers=max(1, args.grok_workers))
    print(
        json.dumps(
            {
                "status": "ok",
                "parser_version": PARSER_VERSION,
                "total_ds3": stats.total_ds3,
                "entry_grok_candidates": stats.entry_grok_candidates,
                "entry_grok_repaired": stats.entry_grok_repaired,
                "grok_candidates": stats.grok_candidates,
                "grok_enriched": stats.grok_enriched,
                "parsed_path": str(PARSED_PATH),
                "chunks_path": str(CHUNKS_PATH),
                "faiss_path": str(FAISS_INDEX_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
