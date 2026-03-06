import os
import json
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_skills_with_grok(job_description: str, api_key: str | None = None) -> list[str]:
    """
    Sends the job description to Grok AI to extract a list of required skills.
    Args:
        job_description: The full text of the job posting.
        api_key: The X.AI API key. If None, looks for it in environment variables.
    Returns:
        A list of skill strings (e.g., ["Python", "AWS", "Machine Learning"]).
    """
    if not api_key:
        api_key = os.getenv("XAI_API_KEY") or st.session_state.get("XAI_API_KEY")

    if not api_key:
        # Fallback/Mock if no key provided yet
        print("[Grok] No API key found. Returning empty list.")
        return []

    try:
        # Grok usually uses an OpenAI-compatible client
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

        prompt = (
            "You are a technical recruiter. Extract a list of the top 15 most "
            "important technical skills, tools, and specific domain keywords "
            "(including company names if specific labs like 'Amazon' or 'Google' are mentioned) "
            "from the following job description. Return ONLY a comma-separated list of keywords. "
            "Do not include any conversational text.\n\n"
            f"Job Description: {job_description}"
        )

        response = client.chat.completions.create(
            model="grok-3", # Found in model list
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts skills from job descriptions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        extracted_text = response.choices[0].message.content
        skills = [s.strip() for s in extracted_text.split(",") if s.strip()]
        return skills

    except Exception as e:
        st.error(f"Error calling Grok AI: {e}")
        return []
def get_explanation_with_grok(job_description: str, candidate_text: str, candidate_name: str, api_key: str | None = None) -> str:
    """
    Sends the job description and candidate resume to Grok AI to explain why the candidate is a top match.
    """
    if not api_key:
        api_key = os.getenv("XAI_API_KEY") or st.session_state.get("XAI_API_KEY")

    if not api_key:
        return "Grok API key missing. Cannot generate explanation."

    try:
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

        prompt = (
            f"You are a technical recruiter. Explain in 2-3 concise sentences why {candidate_name} "
            "is a top-tier candidate for the provided job description. Focus on their most relevant "
            "experience, skills, and past companies. Be specific and professional.\n\n"
            f"Job Description:\n{job_description[:1000]}\n\n"
            f"Candidate Resume Content (Excerpt):\n{candidate_text[:2000]}"
        )

        response = client.chat.completions.create(
            model="grok-3",
            messages=[
                {"role": "system", "content": "You are a professional technical recruiter."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Could not generate Grok explanation: {str(e)}"
