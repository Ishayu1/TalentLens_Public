import os
import json
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_skills_with_grok(job_description: str, api_key: str | None = None) -> list[str]:
    print(f"[Grok Debug] Calling extract_skills_with_grok with model: grok-3")
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
            "You are a strict technical recruiter. Analyze the following job description and extract ONLY "
            "the core technical skills, specific programming languages, frameworks, cloud tools, "
            "and company brand names (like 'Amazon' or 'AWS').\n\n"
            "Rules:\n"
            "1. Return ONLY a comma-separated list of keywords.\n"
            "2. NO generic phrases like 'fast-paced environment', 'autonomy', or 'reliable code'.\n"
            "3. Focus on: Python, Java, C++, LLMs, Reinforcement Learning, AWS, SDE, Intern, etc.\n"
            "4. Max 15 keywords.\n\n"
            f"Job Description: {job_description}"
        )

        response = client.chat.completions.create(
            model="grok-3", # Use Grok-3 Reasoning Model
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
            f"You are a critical technical recruiter. Explain in exactly 2 short bullet points why {candidate_name} "
            "is a top candidate for this role:\n"
            "1. **Core Matching Skills**: List the primary hard technical skills/tools that directly match the job requirements.\n"
            "2. **Relevant Experience**: Highlight specific past internships or technical projects that demonstrate readiness. "
            "DO NOT mention generic college background, leadership, or soft skills unless they are specifically tied to a high-impact technical project.\n\n"
            "Be extremely concise and professional.\n\n"
            f"Job Description:\n{job_description[:1000]}\n\n"
            f"Candidate Resume Content:\n{candidate_text[:2000]}"
        )

        response = client.chat.completions.create(
            model="grok-3", # Use Grok-3 Reasoning Model
            messages=[
                {"role": "system", "content": "You are a professional technical recruiter."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Could not generate Grok explanation: {str(e)}"
