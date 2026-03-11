"""
Reusable UI components for the TalentLens Streamlit app.
Each function renders a self-contained piece of the interface.
"""

from __future__ import annotations

import base64
import html
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


def render_search_bar(selected_skills: list[str]) -> tuple[str, bool, bool, str, str]:
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
        recruiter_company = ""
        recruiter_job_title = ""
        if input_mode == "Job Description":
            col_company, col_title = st.columns(2)
            with col_company:
                recruiter_company = st.text_input(
                    "Company (optional)",
                    placeholder="Optional company override",
                    key="recruiter_company_override",
                )
            with col_title:
                recruiter_job_title = st.text_input(
                    "Job Title (optional)",
                    placeholder="Optional title override",
                    key="recruiter_job_title_override",
                )
        with col_enter:
            search_submitted = st.form_submit_button("Enter", type="primary", use_container_width=True)
        with col_clear:
            clear_clicked = st.form_submit_button("Clear", use_container_width=True)
    return query, search_submitted, clear_clicked, recruiter_company, recruiter_job_title





def _score_color(score: float) -> str:
    if score >= 0.7:
        return COLORS["score_green"]
    if score >= 0.4:
        return COLORS["score_yellow"]
    return COLORS["score_red"]


def _rank_class(rank: int) -> str:
    return f"rank-badge rank-{rank}" if rank <= 3 else "rank-badge"


def _escape_text(value: str) -> str:
    return html.escape(str(value or ""), quote=True)


def _result_card_html(rank: int, name: str, major: str, score_pct: str, score_color: str) -> str:
    # Use different background colors for top 3
    rank_class = f"card-rank-{rank}" if rank <= 3 else "card-rank-4-plus"
    safe_name = _escape_text(name)
    safe_major = _escape_text(major)
    major_line = f'<div class="result-major">{safe_major}</div>' if major else ""
    return (
        f'<div class="result-card {rank_class}">'
        f'<div class="{_rank_class(rank)}">#{rank}</div>'
        f'<div style="flex-grow:1;">'
        f'<div class="result-name">{safe_name}</div>'
        f"{major_line}"
        f"</div>"
        f'<div style="text-align:right;">'
        f'<div class="result-score" style="color:{score_color};">{score_pct}</div>'
        f"</div>"
        f"</div>"
    )


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
        
        # Group card and expander for visual coupling and CSS targeting
        with st.container():
            st.markdown(_result_card_html(r.rank, display_name, r.major, score_pct, score_col), unsafe_allow_html=True)
            with st.expander("Details & Analysis", expanded=False):
                _render_detail_panel(r)


def _render_grok_details(r: ResumeResult):
    grok_status = getattr(r, "grok_status", "") or ""
    if grok_status in ("", "not_requested", "skipped", "unavailable", "error"):
        return
    st.markdown('<div class="evaluation-section">', unsafe_allow_html=True)
    st.markdown("### Resume Evaluation")
    cols = st.columns(2)
    fit = float(getattr(r, "grok_fit_score", 0) or 0)
    quality = float(getattr(r, "grok_resume_quality_score", 0) or 0)
    cols[0].metric("Qualification Fit", f"{fit * 10:.1f}/10")
    cols[1].metric("Resume Quality", f"{quality * 10:.1f}/10")
    
    if getattr(r, "explanation", ""):
        st.markdown(f"**Match Explanation:** {r.explanation}")
    elif getattr(r, "grok_summary", ""):
        st.markdown(f"**Match Explanation:** {r.grok_summary}")
    st.markdown('</div>', unsafe_allow_html=True)


def _render_detail_panel(r: ResumeResult):
    col1, col2 = st.columns(2)
    with col1:
        if r.full_name:
            st.markdown(f"<span class='detail-label'>Name:</span> <span class='detail-value'>{_escape_text(r.full_name)}</span>", unsafe_allow_html=True)
        if r.major:
            st.markdown(f"<span class='detail-label'>Major:</span> <span class='detail-value'>{_escape_text(r.major)}</span>", unsafe_allow_html=True)
        if r.graduation_year:
            st.markdown(f"<span class='detail-label'>Graduation:</span> <span class='detail-value'>{_escape_text(r.graduation_year)}</span>", unsafe_allow_html=True)
    with col2:
        if r.resume_link:
            st.markdown(f"<span class='detail-label'>Resume:</span> <a href='{_escape_text(r.resume_link)}' target='_blank' class='open-link'>Open PDF &rarr;</a>", unsafe_allow_html=True)
        elif r.local_resume_path and Path(r.local_resume_path).exists():
            st.markdown("<span class='detail-label'>Resume:</span> <span class='detail-value'>Local PDF available</span>", unsafe_allow_html=True)
            st.download_button(
                "Download PDF",
                data=Path(r.local_resume_path).read_bytes(),
                file_name=Path(r.local_resume_path).name,
                mime="application/pdf",
                key=f"resume_download_{r.candidate_id or r.filename}",
                use_container_width=False,
            )
        if r.linkedin:
            st.markdown(f"<span class='detail-label'>LinkedIn:</span> <a href='{_escape_text(r.linkedin)}' target='_blank' class='open-link'>Profile &rarr;</a>", unsafe_allow_html=True)
        if r.github:
            st.markdown(f"<span class='detail-label'>GitHub:</span> <a href='{_escape_text(r.github)}' target='_blank' class='open-link'>Profile &rarr;</a>", unsafe_allow_html=True)

    if r.matched_skills:
        skills_html = "".join(f'<span class="matched-skill">{_escape_text(s)}</span>' for s in r.matched_skills)
        st.markdown(f'<div style="margin-top:0.6rem;"><span class="detail-label">Matched Skills:</span><br/>{skills_html}</div>', unsafe_allow_html=True)

    _render_grok_details(r)



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
