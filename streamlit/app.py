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

# ---------------------------------------------------------------------------
# Inject custom CSS
# ---------------------------------------------------------------------------
st.markdown(get_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Initialise search engine (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading search engine...")
def load_engine() -> SearchEngine:
    return SearchEngine()


engine = load_engine()

# ---------------------------------------------------------------------------
# Handle pending actions (must happen *before* widgets are rendered)
# ---------------------------------------------------------------------------
if st.session_state.get("_pending_clear"):
    st.session_state["_pending_clear"] = False
    st.session_state["selected_skills"] = []
    st.session_state["last_results"] = []
    st.session_state["search_query"] = ""

if st.session_state.get("_pending_clear_input"):
    st.session_state["_pending_clear_input"] = False
    st.session_state["search_query"] = ""

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
if "last_results" not in st.session_state:
    st.session_state["last_results"] = []

# ---------------------------------------------------------------------------
# Fixed header bar (always visible at top)
# ---------------------------------------------------------------------------
render_header()

if engine.demo_mode:
    render_demo_banner()

# ---------------------------------------------------------------------------
# Skills panel (collapsible, inline below header)
# ---------------------------------------------------------------------------
selected_skills = render_skills_panel()

# ---------------------------------------------------------------------------
# Sidebar — secondary filters
# ---------------------------------------------------------------------------
grad_years = engine.get_unique_grad_years()
majors = engine.get_unique_majors()
grad_year_filter, major_filter, top_k = render_sidebar_filters(grad_years, majors)

render_sidebar_stats(engine.resume_count, engine.demo_mode)

# ---------------------------------------------------------------------------
# Search bar
# ---------------------------------------------------------------------------
query, search_clicked, clear_clicked = render_search_bar(selected_skills)

if clear_clicked:
    st.session_state["_pending_clear"] = True
    st.rerun()

# ---------------------------------------------------------------------------
# Run search
# ---------------------------------------------------------------------------
input_mode = st.session_state.get("input_mode", "Skills")
has_query = bool(query and query.strip())
has_skills = bool(selected_skills)

if search_clicked:
    if input_mode == "Skills":
        # Parse typed text into individual skills and merge with chip selections
        if has_query:
            typed_skills = [s.strip() for s in query.split(",") if s.strip()]
            for skill in typed_skills:
                if skill not in st.session_state["selected_skills"]:
                    st.session_state["selected_skills"].append(skill)
            # Clear the text input on next rerun so it doesn't duplicate
            st.session_state["_pending_clear_input"] = True

        selected_skills = st.session_state["selected_skills"]
        has_skills = bool(selected_skills)

        if has_skills:
            search_text = ", ".join(selected_skills)
            with st.spinner("Searching resumes..."):
                results = engine.search(
                    query=search_text,
                    top_k=top_k,
                    skill_filters=selected_skills,
                    grad_year_filter=grad_year_filter,
                    major_filter=major_filter,
                )
            st.session_state["last_results"] = results
            if has_query:
                st.rerun()

    elif input_mode == "Job Description" and has_query:
        with st.spinner("Searching resumes..."):
            results = engine.search(
                query=query.strip(),
                top_k=top_k,
                skill_filters=None,
                grad_year_filter=grad_year_filter,
                major_filter=major_filter,
            )
        st.session_state["last_results"] = results

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if st.session_state["last_results"]:
    render_results(st.session_state["last_results"])
elif not has_query and not st.session_state.get("selected_skills"):
    render_initial_state()
