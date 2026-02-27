"""
Custom CSS for the TalentLens Streamlit app.
Reproduces the dark-sidebar / light-content design from the mockup.
"""


def get_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global resets ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main area background ────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1219 100%);
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141824 0%, #0d1117 100%);
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f1f5f9;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {
    color: #cbd5e1;
}

/* ── Sticky header bar ───────────────────────────────────────────────── */
.header-bar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: linear-gradient(135deg, #141824 0%, #0f1219 100%);
    border-bottom: 1px solid #1e293b;
    padding: 0.65rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.header-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0;
    letter-spacing: -0.4px;
    line-height: 1.2;
}

.header-subtitle {
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 0.1rem;
    font-weight: 400;
}

/* Push the main content down so it isn't hidden behind the fixed bar */
.main .block-container {
    padding-top: 5.5rem !important;
    max-width: 1100px;
}

/* ── Skills toggle panel (below header) ──────────────────────────────── */
.skills-toggle-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    color: #cbd5e1;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.skills-toggle-btn:hover {
    background: #1e293b;
    border-color: #475569;
    color: #f1f5f9;
}

.skills-panel {
    background: linear-gradient(135deg, #141824 0%, #0d1117 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}

.skills-panel-title {
    color: #f1f5f9;
    font-size: 0.88rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.skills-panel-hint {
    color: #94a3b8;
    font-size: 0.75rem;
    margin-bottom: 0.5rem;
}

/* ── Demo-mode banner ────────────────────────────────────────────────── */
.demo-banner {
    background: linear-gradient(90deg, #1e293b 0%, #172033 100%);
    border: 1px solid #334155;
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    color: #f1f5f9;
    font-size: 0.88rem;
}

.demo-banner strong {
    color: #f97316;
}

/* ── Search area (st.container with border) ──────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    margin-bottom: 1.5rem;
    border: none !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] input[type="text"] {
    border-radius: 10px;
    border: 1.5px solid #e2e8f0;
    padding: 0.7rem 1rem;
    font-size: 0.95rem;
    background: #f8fafc !important;
    color: #1e293b !important;
    transition: border-color 0.2s;
}

div[data-testid="stVerticalBlockBorderWrapper"] input[type="text"]:focus {
    border-color: #f97316;
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.15);
}

/* Make text inside the search container dark */
div[data-testid="stVerticalBlockBorderWrapper"] label,
div[data-testid="stVerticalBlockBorderWrapper"] .stSelectbox div[data-baseweb="select"] {
    color: #1e293b !important;
}

/* ── Skill chips (sidebar buttons) ───────────────────────────────────── */
.skill-chip {
    display: inline-block;
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
    color: #ffffff;
    padding: 0.45rem 1.1rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.25rem 0;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
    text-align: center;
    border: none;
}

.skill-chip:hover {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
}

.skill-chip-active {
    background: linear-gradient(135deg, #f97316 0%, #fb923c 100%) !important;
    box-shadow: 0 2px 10px rgba(249, 115, 22, 0.35);
}

/* ── Active skill tags in the search area ───────────────────────────── */
.active-skill-tag {
    display: inline-block;
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: #ffffff;
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem 0.3rem 0.2rem 0;
    letter-spacing: 0.2px;
}

/* ── Section heading ─────────────────────────────────────────────────── */
.section-heading {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0.5rem 0 0.75rem 0;
}

/* ── Result card ─────────────────────────────────────────────────────── */
.result-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.15rem 1.4rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    transition: all 0.2s ease;
    border: 1px solid #f1f5f9;
}

.result-card:hover {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.14);
    transform: translateY(-1px);
}

.rank-badge {
    background: linear-gradient(135deg, #0ea5e9 0%, #38bdf8 100%);
    color: #ffffff;
    font-weight: 800;
    font-size: 0.88rem;
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1.1rem;
    flex-shrink: 0;
}

.rank-1 { background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); }
.rank-2 { background: linear-gradient(135deg, #94a3b8 0%, #cbd5e1 100%); color: #1e293b; }
.rank-3 { background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); }

.result-name {
    font-size: 0.98rem;
    font-weight: 600;
    color: #1e293b;
    flex-grow: 1;
}

.result-major {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.15rem;
}

.result-score {
    font-size: 0.82rem;
    font-weight: 700;
    color: #22c55e;
    margin-right: 1.2rem;
    flex-shrink: 0;
}

.open-link {
    color: #2dd4bf;
    font-weight: 600;
    font-size: 0.85rem;
    text-decoration: none;
    flex-shrink: 0;
    transition: color 0.2s;
}

.open-link:hover {
    color: #5eead4;
}

/* ── Expanded detail panel (style the expander content area) ─────────── */
div[data-testid="stExpander"] > details > div[data-testid="stExpanderDetails"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 0 0 10px 10px;
    padding: 1.25rem 1.5rem;
}

.detail-label {
    font-weight: 600;
    color: #94a3b8;
    font-size: 0.82rem;
}

.detail-value {
    color: #f1f5f9;
    font-size: 0.82rem;
}

.matched-skill {
    display: inline-block;
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.15rem 0.25rem;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.text-preview {
    color: #cbd5e1;
    font-size: 0.82rem;
    line-height: 1.6;
    max-height: 120px;
    overflow-y: auto;
    padding: 0.6rem;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 6px;
    border: 1px solid #334155;
    margin-top: 0.5rem;
}

/* ── Stat cards in sidebar ───────────────────────────────────────────── */
.stat-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
    text-align: center;
}

.stat-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #38bdf8;
}

.stat-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Mode indicator ──────────────────────────────────────────────────── */
.mode-label {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

/* ── Empty state ─────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #94a3b8;
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.empty-state-text {
    font-size: 1rem;
    font-weight: 500;
}

.empty-state-hint {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.5rem;
}

/* ── Streamlit overrides ─────────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
    white-space: nowrap;
    min-height: 42px;
}

/* Primary button (Enter) */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #f97316 0%, #fb923c 100%) !important;
    border: none !important;
    color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
    box-shadow: 0 2px 10px rgba(249, 115, 22, 0.35);
}

/* Secondary button (Clear) */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    color: #f1f5f9 !important;
}

.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: #334155 !important;
}

div[data-testid="stExpander"] {
    border: none;
    background: transparent;
}

.stSelectbox label, .stMultiSelect label, .stTextInput label {
    font-weight: 600;
    font-size: 0.85rem;
}

/* Hide Streamlit chrome but keep the sidebar toggle arrow visible */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent !important;
}
div[data-testid="stDecoration"] { display: none; }
/* Hide the "Deploy" button in the toolbar */
div[data-testid="stToolbar"] button[data-testid="stBaseButton-headerNoPadding"] { display: none; }

/* Ensure the sidebar expand/collapse control is always reachable */
section[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}


/* ── Scrollbar styling ───────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
</style>
"""
