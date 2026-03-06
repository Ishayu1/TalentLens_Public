"""
Reusable UI components for the TalentLens Streamlit app.
Each function renders a self-contained piece of the interface.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from config import COLORS, SKILL_SUGGESTIONS
from search import ResumeResult

_LOGO_PATH = Path(__file__).parent / "ds3_logo.png"


def _logo_b64() -> str:
    data = _LOGO_PATH.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def render_header():
    logo_src = _logo_b64() if _LOGO_PATH.exists() else ""
    logo_html = (
        f'<img src="{logo_src}" alt="DS3 Logo" '
        'style="height:38px;width:auto;object-fit:contain;" />'
        if logo_src
        else ""
    )
    st.markdown(
        f"""
        <div class="header-bar">
            <div>
                <div class="header-title">TalentLens</div>
                <div class="header-subtitle">Resume Ranking &bull; Recruiter Dashboard</div>
            </div>
            <div style="display:flex;align-items:center;">
                {logo_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_banner(message: str):
    st.markdown(
        f"""
        <div class="demo-banner">
            <strong>Search Mode Notice</strong> &mdash; {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skills_panel() -> list[str]:
    if "selected_skills" not in st.session_state:
        st.session_state["selected_skills"] = []
    if "skills_panel_open" not in st.session_state:
        st.session_state["skills_panel_open"] = False

    label = "Hide Skill Filters" if st.session_state["skills_panel_open"] else "Show Skill Filters"
    arrow = "\u25B2" if st.session_state["skills_panel_open"] else "\u25BC"
    if st.button(f"{arrow}  {label}", key="skills_panel_toggle"):
        st.session_state["skills_panel_open"] = not st.session_state["skills_panel_open"]
        st.rerun()

    if st.session_state["skills_panel_open"]:
        st.markdown(
            '<div class="skills-panel">'
            '<div class="skills-panel-title">Skill Filters</div>'
            '<div class="skills-panel-hint">Click a skill to add/remove it from your search</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        cols_per_row = 5
        skills_to_show = SKILL_SUGGESTIONS[:15]
        rows = [skills_to_show[i : i + cols_per_row] for i in range(0, len(skills_to_show), cols_per_row)]
        for row in rows:
            cols = st.columns(cols_per_row)
            for col, skill in zip(cols, row):
                is_active = skill in st.session_state["selected_skills"]
                with col:
                    if st.button(
                        skill,
                        key=f"skill_{skill}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        if is_active:
                            st.session_state["selected_skills"].remove(skill)
                        else:
                            st.session_state["selected_skills"].append(skill)
                        st.rerun()

    return st.session_state["selected_skills"]


def render_sidebar_filters(grad_years: list[str], majors: list[str]):
    st.sidebar.markdown("### Filters")
    grad_year = st.sidebar.selectbox("Graduation Year", options=["All"] + grad_years, index=0, key="grad_year_filter")
    major = st.sidebar.selectbox("Major", options=["All"] + majors, index=0, key="major_filter")
    top_k = st.sidebar.slider("Results to show", min_value=5, max_value=50, value=10, step=5, key="top_k")
    return None if grad_year == "All" else grad_year, None if major == "All" else major, top_k


def render_sidebar_stats(resume_count: int, mode_text: str):
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">{resume_count}</div>
            <div class="stat-label">Resumes Indexed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{mode_text}</div>
            <div class="stat-label">Engine Mode</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_search_bar(selected_skills: list[str]) -> tuple[str, bool, bool]:
    if selected_skills:
        tags_html = "".join(f'<span class="active-skill-tag">{s}</span>' for s in selected_skills)
        st.markdown(tags_html, unsafe_allow_html=True)

    _, col_mode_right = st.columns([6, 2.5])
    with col_mode_right:
        input_mode = st.selectbox(
            "Input mode",
            options=["Skills", "Job Description"],
            index=0 if st.session_state.get("input_mode", "Skills") == "Skills" else 1,
            label_visibility="collapsed",
            key="input_mode_widget",
        )
    st.session_state["input_mode"] = input_mode

    placeholder = "e.g. Python, Machine Learning, SQL..." if input_mode == "Skills" else "Paste a job description or requirements..."

    with st.form("search_form", clear_on_submit=False, border=True):
        col_input, col_enter, col_clear = st.columns([5, 1.5, 1.5], vertical_alignment="bottom")
        with col_input:
            query = st.text_input("Search", placeholder=placeholder, label_visibility="collapsed", key="search_query")
        with col_enter:
            search_submitted = st.form_submit_button("Enter", type="primary", use_container_width=True)
        with col_clear:
            clear_clicked = st.form_submit_button("Clear", use_container_width=True)
    return query, search_submitted, clear_clicked


def render_job_description_summary(analysis: dict):
    retrieval = analysis.get("retrieval", {})
    st.markdown("<p class='section-heading'>Parsed Job Description</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Job title:** {analysis.get('job_title') or 'Unknown'}")
        st.markdown(f"**Minimum years:** {analysis.get('minimum_years_experience') if analysis.get('minimum_years_experience') is not None else 'Not detected'}")
        st.markdown(f"**Location:** {analysis.get('location') or 'Not detected'}")
        st.markdown(f"**Remote policy:** {analysis.get('remote_policy') or 'Not detected'}")
    with col2:
        st.markdown(f"**Backend:** {retrieval.get('backend', 'unknown')}")
        st.markdown(f"**Retrieved chunks:** {retrieval.get('retrieved_chunks', 0)}")
        st.markdown(f"**Shortlisted candidates:** {retrieval.get('shortlisted_candidates', 0)}")
        st.markdown(f"**Chunk fetch K:** {retrieval.get('top_k_chunks', 0)}")

    must_have = analysis.get("must_have_skills", [])
    preferred = analysis.get("preferred_skills", [])
    degrees = analysis.get("degree_requirements", [])
    if must_have:
        st.markdown("**Must-have skills**")
        st.markdown(" ".join(f"`{skill}`" for skill in must_have))
    if preferred:
        st.markdown("**Preferred skills**")
        st.markdown(" ".join(f"`{skill}`" for skill in preferred))
    if degrees:
        st.markdown("**Degree requirements**")
        st.markdown(", ".join(degrees))


def _score_color(score: float) -> str:
    if score >= 0.7:
        return COLORS["score_green"]
    if score >= 0.4:
        return COLORS["score_yellow"]
    return COLORS["score_red"]


def _rank_class(rank: int) -> str:
    return f"rank-badge rank-{rank}" if rank <= 3 else "rank-badge"


def render_results(results: list[ResumeResult]):
    if not results:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">&#128270;</div>
                <div class="empty-state-text">No results yet</div>
                <div class="empty-state-hint">
                    Enter a job description or select skills to rank candidates
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown('<p class="section-heading">Ranked Candidates</p>', unsafe_allow_html=True)
    for r in results:
        score_pct = f"{r.score * 100:.0f}%"
        score_col = _score_color(r.score)
        display_name = r.full_name if r.full_name else r.filename
        major_line = f'<div class="result-major">{r.major}</div>' if r.major else ""
        st.markdown(
            f"""
            <div class="result-card">
                <div class="{_rank_class(r.rank)}">#{r.rank}</div>
                <div style="flex-grow:1;">
                    <div class="result-name">{display_name}</div>
                    {major_line}
                </div>
                <div style="text-align:right;">
                    <div class="result-score" style="color:{score_col};">{score_pct}</div>
                    <div style="font-size:0.7rem; opacity:0.6;">Score: {r.score:.3f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Details — {display_name}", expanded=False):
            _render_detail_panel(r)


def _render_filter_status(filter_status: dict):
    if not filter_status:
        return
    st.markdown("### Hard Filters")
    col1, col2, col3 = st.columns(3)
    col1.metric("Must-haves", filter_status.get("must_have_status", "n/a"))
    col2.metric("Years", filter_status.get("years_experience_status", "n/a"))
    col3.metric("Degree", filter_status.get("degree_status", "n/a"))
    if filter_status.get("location_status") != "not_requested":
        st.markdown(f"**Location status:** {filter_status.get('location_status')}")
    matched_must = filter_status.get("matched_must_have_skills", [])
    matched_pref = filter_status.get("matched_preferred_skills", [])
    if matched_must:
        st.markdown("**Matched must-have skills:** " + ", ".join(matched_must))
    if matched_pref:
        st.markdown("**Matched preferred skills:** " + ", ".join(matched_pref))
    if filter_status.get("minimum_years_required") is not None:
        st.markdown(
            f"**Experience:** required {filter_status.get('minimum_years_required')} yrs, "
            f"candidate {filter_status.get('candidate_years_experience', 'unknown')} yrs"
        )


def _render_ranking_details(details: dict):
    if not details:
        return
    st.markdown("### Retrieval Breakdown")
    cols = st.columns(4)
    cols[0].metric("Best evidence", f"{float(details.get('base_search_score', 0.0)):.3f}")
    cols[1].metric("Mean top chunks", f"{float(details.get('mean_top_chunk_score', 0.0)):.3f}")
    cols[2].metric("Evidence chunks", int(details.get("evidence_chunk_count", 0)))
    cols[3].metric("Backend", details.get("retrieval_backend", "n/a"))
    if details.get("total_must_have_count"):
        st.markdown(
            f"**Must-have matches:** {details.get('matched_must_have_count', 0)} / {details.get('total_must_have_count', 0)}"
        )
    if details.get("total_preferred_count"):
        st.markdown(
            f"**Preferred matches:** {details.get('matched_preferred_count', 0)} / {details.get('total_preferred_count', 0)}"
        )


def _render_evidence_chunks(chunks: list[dict]):
    if not chunks:
        return
    st.markdown("### Evidence Chunks")
    for idx, chunk in enumerate(chunks, 1):
        st.markdown(
            f"**{idx}. {chunk.get('section_type', 'other').title()}**  \
Score: `{chunk.get('score', 0.0):.3f}`"
        )
        st.markdown(f"> {chunk.get('text', '').strip()[:700]}")


def _render_detail_panel(r: ResumeResult):
    col1, col2 = st.columns(2)
    with col1:
        if r.full_name:
            st.markdown(f"<span class='detail-label'>Name:</span> <span class='detail-value'>{r.full_name}</span>", unsafe_allow_html=True)
        if r.major:
            st.markdown(f"<span class='detail-label'>Major:</span> <span class='detail-value'>{r.major}</span>", unsafe_allow_html=True)
        if r.graduation_year:
            st.markdown(f"<span class='detail-label'>Graduation:</span> <span class='detail-value'>{r.graduation_year}</span>", unsafe_allow_html=True)
    with col2:
        if r.local_resume_path:
            st.markdown(
                f"<span class='detail-label'>Local Resume:</span> <span class='detail-value'>{Path(r.local_resume_path).name}</span>",
                unsafe_allow_html=True,
            )
        if r.resume_link:
            st.markdown(f"<span class='detail-label'>Resume:</span> <a href='{r.resume_link}' target='_blank' class='open-link'>Open PDF &rarr;</a>", unsafe_allow_html=True)
        if r.linkedin:
            st.markdown(f"<span class='detail-label'>LinkedIn:</span> <a href='{r.linkedin}' target='_blank' class='open-link'>Profile &rarr;</a>", unsafe_allow_html=True)
        if r.github:
            st.markdown(f"<span class='detail-label'>GitHub:</span> <a href='{r.github}' target='_blank' class='open-link'>Profile &rarr;</a>", unsafe_allow_html=True)

    if r.matched_skills:
        skills_html = "".join(f'<span class="matched-skill">{s}</span>' for s in r.matched_skills)
        st.markdown(f'<div style="margin-top:0.6rem;"><span class="detail-label">Matched Skills:</span><br/>{skills_html}</div>', unsafe_allow_html=True)

    _render_filter_status(getattr(r, "hard_filter_status", {}) or {})
    _render_ranking_details(getattr(r, "ranking_details", {}) or {})
    _render_evidence_chunks(getattr(r, "top_evidence_chunks", []) or [])

    if r.local_resume_path:
        pdf_path = Path(r.local_resume_path)
        if pdf_path.exists():
            pdf_bytes = pdf_path.read_bytes()
            pdf_b64 = base64.b64encode(pdf_bytes).decode()
            st.markdown(
                f"""
                <div style="margin-top:0.8rem; margin-bottom:0.5rem; font-size:0.85rem; opacity:0.8;">
                    Resume Preview
                </div>
                <iframe
                    src="data:application/pdf;base64,{pdf_b64}"
                    width="100%"
                    height="900"
                    style="border:1px solid rgba(255,255,255,0.1); border-radius:8px; background:white;"
                ></iframe>
                """,
                unsafe_allow_html=True,
            )

    if r.text_preview:
        st.markdown(f'<div class="text-preview">{r.text_preview}</div>', unsafe_allow_html=True)


def render_initial_state():
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">&#127919;</div>
            <div class="empty-state-text">Welcome to TalentLens</div>
            <div class="empty-state-hint">
                Search by entering a job description, pasting required skills,<br/>
                or clicking skill chips in the sidebar to find your ideal candidates.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
