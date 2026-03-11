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
.dropdown-result {
    margin-bottom: 1.5rem;
}

.result-card {
    background: #ffffff;
    border-radius: 12px 12px 0 0; /* Rounded top only */
    padding: 1.15rem 1.4rem;
    display: flex;
    align-items: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    cursor: pointer;
    position: relative;
    z-index: 2;
}

/* Rank-based background colors */
.card-rank-1 { background-color: #fef3c7 !important; } /* Gold - Amber 100 */
.card-rank-2 { background-color: #e2e8f0 !important; } /* Silver - Slate 200 */
.card-rank-3 { background-color: #ffedd5 !important; } /* Bronze - Orange 100 */
.card-rank-4-plus { background-color: #f0f9ff !important; } /* Light Blue - Sky 50 */

/* Styling the expander to match the card directly above it */
div:has(.card-rank-1) + div[data-testid="stExpander"],
div:has(.card-rank-1) + div[data-testid="stExpander"] > summary {
    background-color: #fef3c7 !important;
}

div:has(.card-rank-2) + div[data-testid="stExpander"],
div:has(.card-rank-2) + div[data-testid="stExpander"] > summary {
    background-color: #e2e8f0 !important;
}

div:has(.card-rank-3) + div[data-testid="stExpander"],
div:has(.card-rank-3) + div[data-testid="stExpander"] > summary {
    background-color: #ffedd5 !important;
}

div:has(.card-rank-4-plus) + div[data-testid="stExpander"],
div:has(.card-rank-4-plus) + div[data-testid="stExpander"] > summary {
    background-color: #f0f9ff !important;
}

.result-card:hover {
    filter: brightness(0.98);
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
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    flex-grow: 1 !important;
}

.result-major {
    font-size: 0.78rem !important;
    color: #64748b !important;
    margin-top: 0.15rem !important;
}

.result-score {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: #16a34a !important; /* Slightly darker green */
    margin-right: 1.2rem !important;
    flex-shrink: 0 !important;
}

.open-link {
    color: #0d9488 !important; /* Darker teal for contrast */
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-decoration: none !important;
    flex-shrink: 0 !important;
    transition: color 0.2s !important;
}

.open-link:hover {
    color: #0f766e !important;
}

/* ── Expanded detail panel (style the expander content area) ─────────── */
/* Seamless expander header */
div[data-testid="stExpander"] {
    border: 1px solid #cbd5e1 !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    background: transparent !important; /* Let sibling-matched color show through */
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

div[data-testid="stExpander"] > summary {
    padding: 0.1rem 1.4rem !important;
    color: #475569 !important; /* Darker blue-gray for better contrast on Light Blue/Gold/Silver */
    font-size: 0.75rem !important;
    background: transparent !important;
    transition: all 0.2s ease;
}

div[data-testid="stExpander"] > summary:hover {
    color: #64748b !important;
}

div[data-testid="stExpander"] > details > div[data-testid="stExpanderDetails"] {
    background: #0f172a !important; /* Navy Blue */
    border: none !important;
    border-radius: 0 0 12px 12px;
    padding: 1.5rem !important;
}

/* Force white text inside the navy expander but preserve metrics and labels */
div[data-testid="stExpanderDetails"] {
    color: #ffffff !important;
}
div[data-testid="stExpanderDetails"] p, 
div[data-testid="stExpanderDetails"] li {
    color: #f1f5f9 !important;
}

/* Except for specific items like matched skills badges which have their own styling */
div[data-testid="stExpanderDetails"] .matched-skill {
    background: rgba(34, 197, 94, 0.2) !important;
    color: #4ade80 !important;
    border: 1px solid rgba(74, 222, 128, 0.3) !important;
}

.evaluation-section {
    margin-top: 1.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.evaluation-section h3 {
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
}

.detail-label {
    font-weight: 600 !important;
    color: #94a3b8 !important; /* Lighter blue-gray for labels on navy */
    font-size: 0.82rem !important;
}

.detail-value {
    color: #f8fafc !important; /* Off-white for values */
    font-size: 0.82rem !important;
}

.matched-skill {
    display: inline-block;
    background: rgba(34, 197, 94, 0.08);
    color: #166534;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0.2rem 0.3rem 0.2rem 0;
    border: 1px solid rgba(22, 101, 52, 0.15);
}

.text-preview {
    color: #475569;
    font-size: 0.82rem;
    line-height: 1.6;
    max-height: 120px;
    overflow-y: auto;
    padding: 0.8rem;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
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
