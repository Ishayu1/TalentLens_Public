"""
TalentLens — Streamlit Recruiter Dashboard

Run from the *streamlit/* directory:
    streamlit run app.py

Or from the project root:
    streamlit run streamlit/app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="TalentLens — Resume Ranking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from components import (
    render_demo_banner,
    render_header,
    render_initial_state,
    render_results,
    render_search_bar,
    render_sidebar_filters,
    render_sidebar_stats,
    render_skills_panel,
)
from search import SearchEngine
from styles import get_css

st.markdown(get_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading search engine...")
def load_engine() -> SearchEngine:
    return SearchEngine(strict_startup=True)


engine = load_engine()


def run_search_with_progress(**search_kwargs):
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0, text="Starting search...")

    def on_progress(progress: float, message: str):
        progress_bar.progress(int(progress * 100), text=message)

    results = engine.search(progress_callback=on_progress, **search_kwargs)
    progress_bar.progress(100, text="Search complete")
    return results

if st.session_state.get("_pending_clear"):
    st.session_state["_pending_clear"] = False
    st.session_state["selected_skills"] = []
    st.session_state["last_results"] = []
    st.session_state["last_job_description_analysis"] = None
    st.session_state["search_query"] = ""
    st.session_state["recruiter_company_override"] = ""
    st.session_state["recruiter_job_title_override"] = ""

if st.session_state.get("_pending_clear_input"):
    st.session_state["_pending_clear_input"] = False
    st.session_state["search_query"] = ""

if "last_results" not in st.session_state:
    st.session_state["last_results"] = []
if "last_job_description_analysis" not in st.session_state:
    st.session_state["last_job_description_analysis"] = None

render_header()

if engine.mode_banner:
    render_demo_banner(engine.mode_banner)

selected_skills = render_skills_panel()

grad_years = engine.get_unique_grad_years()
majors = engine.get_unique_majors()
grad_year_filter, major_filter, top_k = render_sidebar_filters(grad_years, majors)

render_sidebar_stats(engine.resume_count, engine.mode_label)

query, search_clicked, clear_clicked, recruiter_company, recruiter_job_title = render_search_bar(selected_skills)

if clear_clicked:
    st.session_state["_pending_clear"] = True
    st.rerun()

input_mode = st.session_state.get("input_mode", "Skills")
has_query = bool(query and query.strip())

if search_clicked:
    if input_mode == "Skills":
        if has_query:
            typed_skills = [s.strip() for s in query.split(",") if s.strip()]
            for skill in typed_skills:
                if skill not in st.session_state["selected_skills"]:
                    st.session_state["selected_skills"].append(skill)
            st.session_state["_pending_clear_input"] = True

        selected_skills = st.session_state["selected_skills"]
        if selected_skills:
            search_text = ", ".join(selected_skills)
            results = run_search_with_progress(
                query=search_text,
                top_k=top_k,
                skill_filters=selected_skills,
                grad_year_filter=grad_year_filter,
                major_filter=major_filter,
            )
            st.session_state["last_results"] = results
            st.session_state["last_job_description_analysis"] = None
            if has_query:
                st.rerun()

    elif input_mode == "Job Description" and has_query:
        results = run_search_with_progress(
            query=query.strip(),
            top_k=top_k,
            grad_year_filter=grad_year_filter,
            major_filter=major_filter,
            input_mode="Job Description",
            recruiter_company=recruiter_company,
            recruiter_job_title=recruiter_job_title,
        )
        st.session_state["last_results"] = results
        st.session_state["last_job_description_analysis"] = engine.last_query_analysis

# Job description analysis is stored but not rendered directly anymore per user request

if st.session_state["last_results"]:
    render_results(st.session_state["last_results"])
elif not has_query and not st.session_state.get("selected_skills"):
    render_initial_state()
