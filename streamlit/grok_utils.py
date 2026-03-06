import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# xAI docs use "grok-4-1-fast-reasoning"; try hyphenated form first
GROK_MODEL_CANDIDATES = [
    "grok-4-1-fast-reasoning",
    "grok-4.1-fast-reasoning",
]

DEFAULT_RECRUITER_ASSESSMENT = {
    "impact_score": 0.0,
    "technology_fit_score": 0.0,
    "keyword_alignment_score": 0.0,
    "role_fit_score": 0.0,
    "overall_recommendation": "insufficient_signal",
    "evidence": "",
}

DEFAULT_RESUME_RUBRIC_ASSESSMENT = {
    "ats_format_score": 0.0,
    "section_quality_score": 0.0,
    "bullet_quality_score": 0.0,
    "technical_relevance_score": 0.0,
    "truthfulness_score": 0.0,
    "project_strength_score": 0.0,
    "overall_resume_quality_score": 0.0,
    "hard_fail_flags": [],
    "revision_flags": [],
    "strengths": [],
    "risks": [],
    "summary": "",
}


def _resolve_api_key(api_key: str | None = None) -> str | None:
    if api_key:
        return api_key
    return os.getenv("XAI_API_KEY") or st.session_state.get("XAI_API_KEY")


def _build_client(api_key: str | None = None):
    resolved_api_key = _resolve_api_key(api_key)
    if not resolved_api_key:
        return None

    return resolved_api_key


def _report_error(message: str):
    streamlit_error = getattr(st, "error", None)
    if callable(streamlit_error):
        streamlit_error(message)
    else:
        print(message)


def _create_chat_completion(client, messages: list[dict], temperature: float):
    # xAI docs: only model, messages, stream (no temperature in official examples)
    last_error = None
    for model_name in GROK_MODEL_CANDIDATES:
        try:
            response = requests.post(
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
                timeout=120,
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

    return json.loads(cleaned[start : end + 1])


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

def extract_skills_with_grok(job_description: str, api_key: str | None = None) -> list[str]:
    """
    Sends the job description to Grok AI to extract a list of required skills.
    Args:
        job_description: The full text of the job posting.
        api_key: The X.AI API key. If None, looks for it in environment variables.
    Returns:
        A list of skill strings (e.g., ["Python", "AWS", "Machine Learning"]).
    """
    api_key = _resolve_api_key(api_key)

    if not api_key:
        # Fallback/Mock if no key provided yet
        print("[Grok] No API key found. Returning empty list.")
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
        skills = [s.strip() for s in extracted_text.split(",") if s.strip()]
        return skills

    except Exception as e:
        _report_error(f"Error calling Grok AI: {e}")
        return []
def get_explanation_with_grok(job_description: str, candidate_text: str, candidate_name: str, api_key: str | None = None) -> str:
    """
    Sends the job description and candidate resume to Grok AI to explain why the candidate is a top match.
    """
    api_key = _resolve_api_key(api_key)

    if not api_key:
        return "Grok API key missing. Cannot generate explanation."

    try:
        client = _build_client(api_key)

        prompt = (
            f"Write a recruiter-facing match summary for {candidate_name}. In 2-3 concise "
            "sentences, explain how the candidate stacks up against the job description using "
            "this rubric in order: impact and measurable outcomes, relevance of technologies "
            "used, and keyword/role alignment. Sound like a seasoned technical recruiter: "
            "specific, evidence-based, and balanced. If there is a material gap, mention it briefly.\n\n"
            f"Job Description:\n{job_description[:1200]}\n\n"
            f"Candidate Resume Content:\n{candidate_text[:2500]}"
        )

        response = _create_chat_completion(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a seasoned technical recruiter who writes concise, high-signal "
                        "candidate evaluations for hiring managers."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Could not generate Grok explanation: {str(e)}"


def assess_candidate_with_grok(
    job_description: str,
    candidate_text: str,
    candidate_name: str,
    api_key: str | None = None,
) -> dict:
    """
    Returns a structured recruiter assessment used to rerank semantic results.
    """
    api_key = _resolve_api_key(api_key)

    if not api_key:
        return DEFAULT_RECRUITER_ASSESSMENT.copy()

    try:
        client = _build_client(api_key)
        prompt = (
            f"Evaluate {candidate_name} like a seasoned technical recruiter for the job "
            "description below. Score the candidate from 0 to 10 on impact, technology fit, "
            "keyword alignment, and role fit. Favor shipped work, ownership, measurable results, "
            "and direct experience with the job's core stack. Be conservative: do not inflate scores "
            "for weak or implied evidence.\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            '{'
            '"impact_score": number, '
            '"technology_fit_score": number, '
            '"keyword_alignment_score": number, '
            '"role_fit_score": number, '
            '"overall_recommendation": string, '
            '"evidence": string'
            '}\n\n'
            "Keep `overall_recommendation` short, such as strong_match, good_match, mixed_match, "
            "or weak_match. Keep `evidence` to one concise sentence.\n\n"
            f"Job Description:\n{job_description[:1200]}\n\n"
            f"Candidate Resume Content:\n{candidate_text[:3000]}"
        )

        response = _create_chat_completion(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a seasoned technical recruiter who scores resumes against "
                        "job descriptions using evidence from the resume only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        payload = _extract_json_payload(response["choices"][0]["message"]["content"])
        assessment = DEFAULT_RECRUITER_ASSESSMENT.copy()
        assessment["impact_score"] = _coerce_score(payload.get("impact_score"))
        assessment["technology_fit_score"] = _coerce_score(payload.get("technology_fit_score"))
        assessment["keyword_alignment_score"] = _coerce_score(payload.get("keyword_alignment_score"))
        assessment["role_fit_score"] = _coerce_score(payload.get("role_fit_score"))
        assessment["overall_recommendation"] = str(
            payload.get("overall_recommendation", assessment["overall_recommendation"])
        ).strip() or assessment["overall_recommendation"]
        assessment["evidence"] = str(payload.get("evidence", "")).strip()
        return assessment

    except Exception as e:
        return {
            **DEFAULT_RECRUITER_ASSESSMENT,
            "evidence": f"Could not generate recruiter assessment: {str(e)}",
        }


def assess_resume_quality_with_grok(
    job_description: str,
    candidate_text: str,
    candidate_name: str,
    api_key: str | None = None,
) -> dict:
    """
    Returns a structured ATS and resume-quality assessment for reranking.
    """
    api_key = _resolve_api_key(api_key)

    if not api_key:
        return DEFAULT_RESUME_RUBRIC_ASSESSMENT.copy()

    try:
        client = _build_client(api_key)
        prompt = (
            f"Score {candidate_name}'s resume against the job description and this resume-review rubric. "
            "Assume the candidate is an early-career software or CS applicant unless the resume clearly shows otherwise. "
            "Judge from the resume text only and be conservative.\n\n"
            "Rubric priorities:\n"
            "- one-page, ATS-friendly, reverse-chronological, standard-heading resume\n"
            "- strong technical skills section near the top\n"
            "- education near the top for students\n"
            "- bullets should be action + work + result, with quantification whenever supported\n"
            "- projects should demonstrate real technical depth, ownership, deployment, users, performance, or measurable impact\n"
            "- content should be tailored to the job description using truthful keywords\n"
            "- unsupported or exaggerated claims are a serious risk\n\n"
            "Hard-fail categories to detect when evidence supports them:\n"
            "- ats_format_risk\n"
            "- not_tailored_to_role\n"
            "- unsupported_or_exaggerated_skills\n\n"
            "Revision flag categories to detect when evidence supports them:\n"
            "- weak_or_unquantified_bullets\n"
            "- irrelevant_content_crowding_out_technical_evidence\n"
            "- missing_or_weak_projects\n"
            "- weak_skills_section\n"
            "- weak_section_order_or_missing_standard_headings\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            "{"
            "\"ats_format_score\": number, "
            "\"section_quality_score\": number, "
            "\"bullet_quality_score\": number, "
            "\"technical_relevance_score\": number, "
            "\"truthfulness_score\": number, "
            "\"project_strength_score\": number, "
            "\"overall_resume_quality_score\": number, "
            "\"hard_fail_flags\": string[], "
            "\"revision_flags\": string[], "
            "\"strengths\": string[], "
            "\"risks\": string[], "
            "\"summary\": string"
            "}\n\n"
            "Scoring guidance:\n"
            "- 9-10 means excellent evidence\n"
            "- 7-8 means strong with minor gaps\n"
            "- 5-6 means mixed or inconsistent\n"
            "- 3-4 means weak\n"
            "- 0-2 means severe issue or missing evidence\n\n"
            "Keep strengths and risks to at most 3 short items each. Keep summary to 1-2 concise sentences.\n\n"
            f"Job Description:\n{job_description[:1500]}\n\n"
            f"Candidate Resume Content:\n{candidate_text[:5000]}"
        )

        response = _create_chat_completion(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a seasoned technical recruiter and ATS resume reviewer. "
                        "Score resume quality, tailoring, and risk signals using only evidence present in the resume text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        payload = _extract_json_payload(response["choices"][0]["message"]["content"])
        assessment = DEFAULT_RESUME_RUBRIC_ASSESSMENT.copy()
        assessment["ats_format_score"] = _coerce_score(payload.get("ats_format_score"))
        assessment["section_quality_score"] = _coerce_score(payload.get("section_quality_score"))
        assessment["bullet_quality_score"] = _coerce_score(payload.get("bullet_quality_score"))
        assessment["technical_relevance_score"] = _coerce_score(payload.get("technical_relevance_score"))
        assessment["truthfulness_score"] = _coerce_score(payload.get("truthfulness_score"))
        assessment["project_strength_score"] = _coerce_score(payload.get("project_strength_score"))
        assessment["overall_resume_quality_score"] = _coerce_score(payload.get("overall_resume_quality_score"))
        assessment["hard_fail_flags"] = _coerce_list(payload.get("hard_fail_flags"))
        assessment["revision_flags"] = _coerce_list(payload.get("revision_flags"))
        assessment["strengths"] = _coerce_list(payload.get("strengths"))
        assessment["risks"] = _coerce_list(payload.get("risks"))
        assessment["summary"] = str(payload.get("summary", "")).strip()
        return assessment

    except Exception as e:
        return {
            **DEFAULT_RESUME_RUBRIC_ASSESSMENT,
            "risks": [f"Could not generate resume rubric assessment: {str(e)}"],
        }
