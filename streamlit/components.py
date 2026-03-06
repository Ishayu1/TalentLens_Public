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
    """Return the DS3 logo as a base64 data-URI for use in raw HTML."""
    data = _LOGO_PATH.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


# ---------------------------------------------------------------------------
# Header (fixed bar at top of page)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Engine-mode banner
# ---------------------------------------------------------------------------
def render_demo_banner(message: str):
    st.markdown(
        f"""
        <div class="demo-banner">
            <strong>Search Mode Notice</strong> &mdash; {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Inline skills panel (collapsible, below header)
# ---------------------------------------------------------------------------
def render_skills_panel() -> list[str]:
    """
    Render a toggle button + collapsible skill-chip panel in the main
    content area (directly below the fixed header).  Returns the list
    of currently selected skills.
    """
    if "selected_skills" not in st.session_state:
        st.session_state["selected_skills"] = []
    if "skills_panel_open" not in st.session_state:
        st.session_state["skills_panel_open"] = False

    # Toggle button
    label = "Hide Skill Filters" if st.session_state["skills_panel_open"] else "Show Skill Filters"
    arrow = "\u25B2" if st.session_state["skills_panel_open"] else "\u25BC"
    if st.button(f"{arrow}  {label}", key="skills_panel_toggle"):
        st.session_state["skills_panel_open"] = not st.session_state["skills_panel_open"]
        st.rerun()

    # Panel contents
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
        rows = [
            skills_to_show[i : i + cols_per_row]
            for i in range(0, len(skills_to_show), cols_per_row)
        ]
        for row in rows:
            cols = st.columns(cols_per_row)
            for col, skill in zip(cols, row):
                is_active = skill in st.session_state["selected_skills"]
                btn_type = "primary" if is_active else "secondary"
                with col:
                    if st.button(
                        skill,
                        key=f"skill_{skill}",
                        use_container_width=True,
                        type=btn_type,
                    ):
                        if is_active:
                            st.session_state["selected_skills"].remove(skill)
                        else:
                            st.session_state["selected_skills"].append(skill)
                        st.rerun()

    return st.session_state["selected_skills"]


# ---------------------------------------------------------------------------
# Sidebar — filters (graduation year, major, top-k)
# ---------------------------------------------------------------------------
def render_sidebar_filters(grad_years: list[str], majors: list[str]):
    st.sidebar.markdown("### Filters")

    grad_year = st.sidebar.selectbox(
        "Graduation Year",
        options=["All"] + grad_years,
        index=0,
        key="grad_year_filter",
    )

    major = st.sidebar.selectbox(
        "Major",
        options=["All"] + majors,
        index=0,
        key="major_filter",
    )

    top_k = st.sidebar.slider(
        "Results to show",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        key="top_k",
    )

    return (
        None if grad_year == "All" else grad_year,
        None if major == "All" else major,
        top_k,
    )


# ---------------------------------------------------------------------------
# Sidebar — stats
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Search bar + action buttons
# ---------------------------------------------------------------------------
def render_search_bar(selected_skills: list[str]) -> tuple[str, bool, bool]:
    """
    Returns (query_text, search_submitted, clear_clicked).

    The dropdown toggles between two input modes:
      * **Skills** — the text field accepts comma-separated skills and
        sidebar skill chips are active.
      * **Job Description** — the text field accepts free-form text
        (a full job posting, paragraph, etc.).

    The mode selector lives *outside* the form so switching modes
    triggers an instant rerun (updates the placeholder immediately).
    """
    if selected_skills:
        tags_html = "".join(
            f'<span class="active-skill-tag">{s}</span>' for s in selected_skills
        )
        st.markdown(tags_html, unsafe_allow_html=True)

    # --- mode selector (outside form → instant rerun on change) ----------
    _, col_mode_right = st.columns([6, 2.5])
    with col_mode_right:
        input_mode = st.selectbox(
            "Input mode",
            options=["Skills", "Job Description"],
            index=(
                0
                if st.session_state.get("input_mode", "Skills") == "Skills"
                else 1
            ),
            label_visibility="collapsed",
            key="input_mode_widget",
        )
    st.session_state["input_mode"] = input_mode

    placeholder = (
        "e.g. Python, Machine Learning, SQL..."
        if input_mode == "Skills"
        else "Paste a job description or requirements..."
    )

    # --- search form (Enter key + button both submit) --------------------
    with st.form("search_form", clear_on_submit=False, border=True):
        col_input, col_enter, col_clear = st.columns(
            [5, 1.5, 1.5], vertical_alignment="bottom"
        )

        with col_input:
            query = st.text_input(
                "Search",
                placeholder=placeholder,
                label_visibility="collapsed",
                key="search_query",
            )

        with col_enter:
            search_submitted = st.form_submit_button(
                "Enter",
                type="primary",
                use_container_width=True,
            )

        with col_clear:
            clear_clicked = st.form_submit_button(
                "Clear",
                use_container_width=True,
            )

    return query, search_submitted, clear_clicked


# ---------------------------------------------------------------------------
# Result cards
# ---------------------------------------------------------------------------
def _score_color(score: float) -> str:
    if score >= 0.7:
        return COLORS["score_green"]
    if score >= 0.4:
        return COLORS["score_yellow"]
    return COLORS["score_red"]


def _rank_class(rank: int) -> str:
    if rank <= 3:
        return f"rank-badge rank-{rank}"
    return "rank-badge"


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
        rank_cls = _rank_class(r.rank)

        display_name = r.full_name if r.full_name else r.filename
        major_line = f'<div class="result-major">{r.major}</div>' if r.major else ""

        card_html = f"""
        <div class="result-card">
            <div class="{rank_cls}">#{r.rank}</div>
            <div style="flex-grow:1;">
                <div class="result-name">{display_name}</div>
                {major_line}
            </div>
            <div style="text-align:right;">
                <div class="result-score" style="color:{score_col};">{score_pct}</div>
                <div style="font-size:0.7rem; opacity:0.6;">Score: {r.score:.3f}</div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        with st.expander(f"Details — {display_name}", expanded=False):
            _render_detail_panel(r)


def _render_metric_section(
    title: str,
    label_to_key_weight: dict[str, tuple[str, int]],
    breakdown: dict,
) -> None:
    """Render a subsection of scores (e.g. job match or resume quality) with weights."""
    if not breakdown:
        return
    st.markdown(f"**{title}**")
    # Scores from Grok are 0-10
    for label, (key, weight_pct) in label_to_key_weight.items():
        raw = breakdown.get(key)
        try:
            val = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            val = None
        if val is not None:
            score_10 = max(0.0, min(10.0, val))
            bar_color = COLORS["score_green"] if score_10 >= 7 else (COLORS["score_yellow"] if score_10 >= 4 else COLORS["score_red"])
            st.markdown(
                f'<div style="margin-bottom:0.35rem; font-size:0.9rem;">'
                f'<span style="opacity:0.9;">{label}</span> '
                f'<span style="color:{bar_color}; font-weight:600;">{score_10:.1f}/10</span> '
                f'<span style="opacity:0.6; font-size:0.8rem;">(weight {weight_pct}%)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("")  # spacing


def _render_ranking_formula(details: dict) -> None:
    if not details:
        return

    st.markdown("#### Ranking Formula")
    formula_cols = st.columns(4)
    formula_cols[0].metric("Base Search", f"{float(details.get('base_search_score', 0.0)):.2f}/10")

    if details.get("job_match_score") is not None:
        formula_cols[1].metric("Job Match", f"{float(details.get('job_match_score', 0.0)):.2f}/10")
    else:
        formula_cols[1].metric("Job Match", "N/A")

    if details.get("resume_quality_score") is not None:
        formula_cols[2].metric("Resume Quality", f"{float(details.get('resume_quality_score', 0.0)):.2f}/10")
    else:
        formula_cols[2].metric("Resume Quality", "N/A")

    if details.get("penalty_applied") is not None:
        formula_cols[3].metric("Penalty", f"-{float(details.get('penalty_applied', 0.0)):.2f}/10")
    else:
        formula_cols[3].metric("Penalty", "0.00/10")

    mode = details.get("mode", "")
    if mode:
        st.caption(f"Ranking mode: `{mode}`")

    matched_skill_count = details.get("matched_skill_count")
    matched_skill_ratio = details.get("matched_skill_ratio")
    if matched_skill_count is not None:
        ratio_text = ""
        if matched_skill_ratio is not None:
            ratio_text = f" ({float(matched_skill_ratio) * 100:.0f}% of query skills)"
        st.markdown(f"**Matched skills used in ranking:** {int(matched_skill_count)}{ratio_text}")

    hard_fail_count = details.get("hard_fail_count")
    revision_flag_count = details.get("revision_flag_count")
    if hard_fail_count is not None or revision_flag_count is not None:
        st.markdown(
            f"**Penalty drivers:** {int(hard_fail_count or 0)} hard-fail flags, "
            f"{int(revision_flag_count or 0)} revision flags"
        )

def _render_detail_panel(r: ResumeResult):
    col1, col2 = st.columns(2)
    with col1:
        if r.full_name:
            st.markdown(f'<span class="detail-label">Name:</span> <span class="detail-value">{r.full_name}</span>', unsafe_allow_html=True)
        if r.major:
            st.markdown(f'<span class="detail-label">Major:</span> <span class="detail-value">{r.major}</span>', unsafe_allow_html=True)
        if r.graduation_year:
            st.markdown(f'<span class="detail-label">Graduation:</span> <span class="detail-value">{r.graduation_year}</span>', unsafe_allow_html=True)

    with col2:
        if r.local_resume_path:
            st.markdown(
                f'<span class="detail-label">Local Resume:</span> <span class="detail-value">{Path(r.local_resume_path).name}</span>',
                unsafe_allow_html=True,
            )
        if r.resume_link:
            st.markdown(f'<span class="detail-label">Resume:</span> <a href="{r.resume_link}" target="_blank" class="open-link">Open PDF &rarr;</a>', unsafe_allow_html=True)
        if r.linkedin:
            st.markdown(f'<span class="detail-label">LinkedIn:</span> <a href="{r.linkedin}" target="_blank" class="open-link">Profile &rarr;</a>', unsafe_allow_html=True)
        if r.github:
            st.markdown(f'<span class="detail-label">GitHub:</span> <a href="{r.github}" target="_blank" class="open-link">Profile &rarr;</a>', unsafe_allow_html=True)

    if r.local_resume_path:
        pdf_path = Path(r.local_resume_path)
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

    if r.matched_skills:
        skills_html = "".join(f'<span class="matched-skill">{s}</span>' for s in r.matched_skills)
        st.markdown(
            f'<div style="margin-top:0.6rem;"><span class="detail-label">Matched Skills:</span><br/>{skills_html}</div>',
            unsafe_allow_html=True,
        )

    ranking_details = getattr(r, "ranking_details", {}) or {}
    if ranking_details or r.recruiter_score or r.resume_quality_score or r.semantic_score:
        st.markdown("### Score Breakdown")
        score_cols = st.columns(4)
        score_cols[0].metric("Combined Score", f"{r.score:.3f}")
        score_cols[1].metric("Base Search", f"{r.semantic_score:.3f}")
        score_cols[2].metric("Job Match", f"{r.recruiter_score:.1f}/10")
        score_cols[3].metric("Resume Quality", f"{r.resume_quality_score:.1f}/10")

        _render_ranking_formula(ranking_details)

        # Section-based metrics that explain why the resume is ranked this way
        recruiter_breakdown = getattr(r, "recruiter_breakdown", {}) or {}
        quality_breakdown = getattr(r, "resume_quality_breakdown", {}) or {}
        if recruiter_breakdown or quality_breakdown:
            st.markdown("#### Why this rank — metrics by section")
            _render_metric_section(
                "Job match (vs job description)",
                {
                    "Impact": ("impact_score", 35),
                    "Technology fit": ("technology_fit_score", 35),
                    "Keyword alignment": ("keyword_alignment_score", 20),
                    "Role fit": ("role_fit_score", 10),
                },
                recruiter_breakdown,
            )
            _render_metric_section(
                "Resume quality (ATS & content)",
                {
                    "ATS format": ("ats_format_score", 18),
                    "Section quality": ("section_quality_score", 12),
                    "Bullet quality": ("bullet_quality_score", 20),
                    "Technical relevance": ("technical_relevance_score", 22),
                    "Truthfulness": ("truthfulness_score", 18),
                    "Project strength": ("project_strength_score", 10),
                },
                quality_breakdown,
            )

    if r.hard_fail_flags:
        st.error("Hard-fail risks: " + ", ".join(r.hard_fail_flags))

    if r.resume_flags:
        st.warning("Revision flags: " + ", ".join(r.resume_flags))

    rubric = getattr(r, "resume_quality_breakdown", {}) or {}
    if rubric:
        strengths = rubric.get("strengths", [])
        risks = rubric.get("risks", [])
        summary = rubric.get("summary", "")
        if summary:
            st.markdown(f"**Resume Summary:** {summary}")
        if strengths:
            st.markdown("**Strengths**")
            for item in strengths:
                st.markdown(f"- {item}")
        if risks:
            st.markdown("**Risks**")
            for item in risks:
                st.markdown(f"- {item}")

    explanation = getattr(r, "explanation", "")
    if explanation:
        st.markdown(
            f"""
            <div style="margin-top:1rem; padding:0.8rem; background:rgba(121, 40, 202, 0.1); border-left:3px solid #7928ca; border-radius:4px;">
                <div style="font-size:0.8rem; font-weight:600; color:#7928ca; margin-bottom:0.3rem;">
                    ✨ Grok AI Match Analysis
                </div>
                <div style="font-size:0.9rem; font-style:italic; line-height:1.4;">
                    "{r.explanation}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if r.text_preview:
        st.markdown(f'<div class="text-preview">{r.text_preview}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Empty / initial state
# ---------------------------------------------------------------------------
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
