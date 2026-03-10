from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from copy import deepcopy
from threading import Lock
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

GROK_MODEL_CANDIDATES = [
    "grok-4-1-fast-reasoning",
    "grok-4.1-fast-reasoning",
]

DEFAULT_CANDIDATE_PACKET_ASSESSMENT = {
    "status": "unavailable",
    "qualification_match_score": 0.0,
    "company_relevance_score": 0.0,
    "experience_relevance_score": 0.0,
    "bullet_quality_score": 0.0,
    "project_strength_score": 0.0,
    "resume_quality_score": 0.0,
    "matched_requirements": [],
    "missing_requirements": [],
    "weakness_flags": [],
    "summary": "",
}

_ASSESSMENT_REQUIRED_KEYS = {
    "qualification_match_score",
    "company_relevance_score",
    "experience_relevance_score",
    "bullet_quality_score",
    "project_strength_score",
    "resume_quality_score",
    "matched_requirements",
    "missing_requirements",
    "weakness_flags",
    "summary",
}
_ASSESSMENT_CACHE: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
_ASSESSMENT_CACHE_SIZE = 256
_ASSESSMENT_CACHE_LOCK = Lock()
_THREAD_LOCAL = threading.local()


def _resolve_api_key(api_key: str | None = None) -> str | None:
    if api_key:
        return api_key
    session_state = getattr(st, "session_state", None)
    if session_state is not None:
        value = session_state.get("XAI_API_KEY")
        if value:
            return value
    return os.getenv("XAI_API_KEY")


def has_grok_api_key(api_key: str | None = None) -> bool:
    return bool(_resolve_api_key(api_key))


def _build_client(api_key: str | None = None) -> str | None:
    return _resolve_api_key(api_key)


def _get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def _report_error(message: str):
    streamlit_error = getattr(st, "error", None)
    if callable(streamlit_error):
        streamlit_error(message)
    else:
        print(message)


def _create_chat_completion(client: str, messages: list[dict], temperature: float):
    del temperature
    last_error = None
    session = _get_session()
    for model_name in GROK_MODEL_CANDIDATES:
        try:
            response = session.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {client}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                },
                timeout=45,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            last_error = exc
            err_body = ""
            try:
                err_body = (exc.response.text or "")[:500]
            except Exception:
                pass
            if "Model not found" not in str(exc) and "model" not in err_body.lower():
                raise
        except Exception as exc:
            last_error = exc
            if "Model not found" not in str(exc):
                raise

    raise last_error


def _extract_json_payload(raw_text: str) -> dict:
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
        raise ValueError("No JSON object found in Grok response.")

    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Grok response JSON must be an object.")
    return payload


def _coerce_score(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(10.0, numeric))


def _coerce_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_assessment(payload: dict[str, Any]) -> dict[str, Any]:
    missing_keys = sorted(_ASSESSMENT_REQUIRED_KEYS - payload.keys())
    if missing_keys:
        raise ValueError(f"Missing keys in Grok response: {', '.join(missing_keys)}")

    assessment = deepcopy(DEFAULT_CANDIDATE_PACKET_ASSESSMENT)
    assessment["status"] = "ok"
    assessment["qualification_match_score"] = _coerce_score(payload.get("qualification_match_score"))
    assessment["company_relevance_score"] = _coerce_score(payload.get("company_relevance_score"))
    assessment["experience_relevance_score"] = _coerce_score(payload.get("experience_relevance_score"))
    assessment["bullet_quality_score"] = _coerce_score(payload.get("bullet_quality_score"))
    assessment["project_strength_score"] = _coerce_score(payload.get("project_strength_score"))
    assessment["resume_quality_score"] = _coerce_score(payload.get("resume_quality_score"))
    assessment["matched_requirements"] = _coerce_list(payload.get("matched_requirements"))
    assessment["missing_requirements"] = _coerce_list(payload.get("missing_requirements"))
    assessment["weakness_flags"] = _coerce_list(payload.get("weakness_flags"))
    assessment["summary"] = str(payload.get("summary", "")).strip()
    return assessment


def _assessment_cache_key(job_description: str, candidate_id: str) -> tuple[str, str]:
    digest = hashlib.sha1(job_description.encode("utf-8")).hexdigest()
    return digest, candidate_id


def _read_cached_assessment(job_description: str, candidate_id: str) -> dict[str, Any] | None:
    key = _assessment_cache_key(job_description, candidate_id)
    with _ASSESSMENT_CACHE_LOCK:
        cached = _ASSESSMENT_CACHE.get(key)
        if cached is None:
            return None
        _ASSESSMENT_CACHE.move_to_end(key)
        return deepcopy(cached)


def _store_cached_assessment(job_description: str, candidate_id: str, assessment: dict[str, Any]) -> None:
    key = _assessment_cache_key(job_description, candidate_id)
    with _ASSESSMENT_CACHE_LOCK:
        _ASSESSMENT_CACHE[key] = deepcopy(assessment)
        _ASSESSMENT_CACHE.move_to_end(key)
        while len(_ASSESSMENT_CACHE) > _ASSESSMENT_CACHE_SIZE:
            _ASSESSMENT_CACHE.popitem(last=False)


def extract_skills_with_grok(job_description: str, api_key: str | None = None) -> list[str]:
    api_key = _resolve_api_key(api_key)
    if not api_key:
        return []

    try:
        client = _build_client(api_key)
        prompt = (
            "Review the following job description like a seasoned technical recruiter. "
            "Extract the most resume-searchable signals that separate strong candidates: "
            "must-have technologies, important frameworks, domain keywords, role titles, "
            "and concrete impact or production terms that are likely to appear on resumes. "
            "Prioritize the strongest 12 to 18 terms. Include company names only when that "
            "background is explicitly relevant. Exclude generic soft skills and filler. "
            "Return ONLY a comma-separated list with no bullets, numbering, or commentary.\n\n"
            f"Job Description: {job_description}"
        )

        response = _create_chat_completion(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a seasoned technical recruiter for high-bar software, "
                        "data, and AI roles. Extract concise recruiter-grade search terms "
                        "that help rank resumes against the job description."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        extracted_text = response["choices"][0]["message"]["content"]
        return [skill.strip() for skill in extracted_text.split(",") if skill.strip()]
    except Exception as exc:
        _report_error(f"Error calling Grok AI: {exc}")
        return []


def assess_candidate_packet_with_grok(
    job_description: str,
    parsed_requirements: dict[str, Any],
    candidate_packet: dict[str, Any],
    api_key: str | None = None,
) -> dict[str, Any]:
    candidate_id = str(candidate_packet.get("candidate_id", "")).strip()
    if candidate_id:
        cached = _read_cached_assessment(job_description, candidate_id)
        if cached is not None:
            return cached

    api_key = _resolve_api_key(api_key)
    if not api_key:
        return deepcopy(DEFAULT_CANDIDATE_PACKET_ASSESSMENT)

    try:
        client = _build_client(api_key)
        prompt = (
            "Score this early-career technical candidate against the job description. "
            "Judge only from resume evidence. Do not invent facts. When a requirement "
            "cannot be confirmed, treat it as unclear and place it in missing_requirements "
            "with an 'unclear:' prefix rather than assuming it is met.\n\n"
            "Resume review rubric:\n"
            "- Experience bullets should begin with strong action language, stay concise, and emphasize accomplishments.\n"
            "- Strong bullets follow action + what was built or improved + how it was done + result or impact.\n"
            "- Quantified outcomes are stronger than vague claims.\n"
            "- Project entries should show project name, technologies, and 2 to 4 meaningful bullets.\n"
            "- Projects are stronger when they show deployment, real users, measurable performance, open-source contribution, or clear ownership.\n"
            "- Weak bullets include phrases like worked on, helped with, or responsible for without technical substance or outcome.\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            "{"
            "\"qualification_match_score\": number, "
            "\"company_relevance_score\": number, "
            "\"experience_relevance_score\": number, "
            "\"bullet_quality_score\": number, "
            "\"project_strength_score\": number, "
            "\"resume_quality_score\": number, "
            "\"matched_requirements\": string[], "
            "\"missing_requirements\": string[], "
            "\"weakness_flags\": string[], "
            "\"summary\": string"
            "}\n\n"
            "Use 0 to 10 scoring where 10 is excellent evidence and 0 is absent or severely weak.\n\n"
            f"Parsed job requirements:\n{json.dumps(parsed_requirements, ensure_ascii=True)}\n\n"
            f"Candidate packet:\n{json.dumps(candidate_packet, ensure_ascii=True)}\n\n"
            f"Original job description:\n{job_description[:2500]}"
        )

        response = _create_chat_completion(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous technical recruiter. Score candidates conservatively, "
                        "ground everything in resume evidence, and avoid guessing."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        payload = _extract_json_payload(response["choices"][0]["message"]["content"])
        assessment = _normalize_assessment(payload)
        if candidate_id:
            _store_cached_assessment(job_description, candidate_id, assessment)
        return assessment
    except Exception as exc:
        return {
            **deepcopy(DEFAULT_CANDIDATE_PACKET_ASSESSMENT),
            "status": "error",
            "summary": f"Could not generate Grok assessment: {exc}",
        }
