"""
CinePredict — Box Office Revenue Forecaster
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CinePredict",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-color: #0C0C0F;
    color: #E8E6E0;
}
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] {
    background: #0C0C0F;
    border-right: 1px solid #1e1e28;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Nav Bar ── */
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(12,12,15,0.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid #1e1e28;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 48px;
    height: 64px;
}
.nav-logo {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: #E8E6E0;
}
.nav-logo span { color: #E8B84B; }
.nav-links {
    display: flex;
    gap: 4px;
}
.nav-btn {
    background: none;
    border: none;
    color: #888;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    padding: 8px 18px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.2px;
}
.nav-btn:hover { color: #E8E6E0; background: #1a1a22; }
.nav-btn.active { color: #E8B84B; background: rgba(232,184,75,0.1); }

/* ── Page wrapper ── */
.page { padding: 0 48px 80px; }

/* ── Hero section ── */
.hero {
    min-height: 88vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 80px 20px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(232,184,75,0.07) 0%, transparent 70%);
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #E8B84B;
    margin-bottom: 24px;
    opacity: 0.9;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(56px, 8vw, 96px);
    font-weight: 900;
    line-height: 0.95;
    color: #E8E6E0;
    margin-bottom: 8px;
    letter-spacing: -2px;
}
.hero-title-accent {
    font-family: 'Playfair Display', serif;
    font-size: clamp(56px, 8vw, 96px);
    font-weight: 900;
    line-height: 0.95;
    color: #E8B84B;
    letter-spacing: -2px;
    display: block;
}
.hero-sub {
    font-size: 18px;
    color: #666;
    font-weight: 300;
    max-width: 500px;
    line-height: 1.7;
    margin: 28px auto 0;
    letter-spacing: 0.2px;
}
.hero-stats {
    display: flex;
    gap: 48px;
    margin-top: 60px;
    padding-top: 48px;
    border-top: 1px solid #1e1e28;
}
.stat-item { text-align: center; }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    font-weight: 700;
    color: #E8B84B;
    line-height: 1;
}
.stat-label {
    font-size: 12px;
    color: #555;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 6px;
    font-family: 'DM Mono', monospace;
}

/* ── Feature cards ── */
.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: #1a1a22;
    border: 1px solid #1a1a22;
    border-radius: 16px;
    overflow: hidden;
    margin: 64px 0;
}
.feat-card {
    background: #0C0C0F;
    padding: 40px 32px;
    transition: background 0.2s;
}
.feat-card:hover { background: #111116; }
.feat-icon {
    font-size: 28px;
    margin-bottom: 20px;
    display: block;
}
.feat-title {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: #E8E6E0;
    margin-bottom: 10px;
}
.feat-desc {
    font-size: 14px;
    color: #555;
    line-height: 1.8;
}

/* ── Section heading ── */
.section-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #E8B84B;
    margin-bottom: 12px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 40px;
    font-weight: 700;
    color: #E8E6E0;
    line-height: 1.1;
    margin-bottom: 16px;
    letter-spacing: -0.5px;
}
.section-desc {
    font-size: 16px;
    color: #555;
    line-height: 1.8;
    max-width: 520px;
}

/* ── How it works ── */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 24px;
    margin-top: 48px;
}
.step-card {
    position: relative;
    padding: 32px 24px;
    border: 1px solid #1a1a22;
    border-radius: 12px;
    background: #0C0C0F;
}
.step-num {
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    font-weight: 900;
    color: #1a1a22;
    line-height: 1;
    margin-bottom: 12px;
}
.step-title {
    font-size: 15px;
    font-weight: 600;
    color: #E8E6E0;
    margin-bottom: 8px;
}
.step-desc { font-size: 13px; color: #555; line-height: 1.7; }

/* ── CTA banner ── */
.cta-banner {
    background: linear-gradient(135deg, #1a1505 0%, #0C0C0F 60%);
    border: 1px solid #2a2010;
    border-radius: 20px;
    padding: 64px 48px;
    text-align: center;
    margin: 40px 0;
    position: relative;
    overflow: hidden;
}
.cta-banner::before {
    content: '';
    position: absolute;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(232,184,75,0.08), transparent 70%);
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
}
.cta-title {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    font-weight: 900;
    color: #E8E6E0;
    margin-bottom: 16px;
    position: relative;
}
.cta-sub { font-size: 16px; color: #666; margin-bottom: 32px; position: relative; }

/* ── Prediction page ── */
.pred-header {
    padding: 56px 0 40px;
    border-bottom: 1px solid #1a1a22;
    margin-bottom: 40px;
}
.pred-title {
    font-family: 'Playfair Display', serif;
    font-size: 48px;
    font-weight: 900;
    color: #E8E6E0;
    letter-spacing: -1px;
    line-height: 1;
}
.pred-sub { font-size: 15px; color: #555; margin-top: 10px; }

.form-section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #E8B84B;
    margin: 32px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a1a22;
}

/* Input overrides */
.stTextInput input, .stNumberInput input {
    background: #111116 !important;
    border: 1px solid #1e1e28 !important;
    border-radius: 10px !important;
    color: #E8E6E0 !important;
    font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 12px 14px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #E8B84B !important;
    box-shadow: 0 0 0 2px rgba(232,184,75,0.12) !important;
}
.stSelectbox > div > div {
    background: #111116 !important;
    border: 1px solid #1e1e28 !important;
    border-radius: 10px !important;
    color: #E8E6E0 !important;
}
.stMultiSelect > div > div {
    background: #111116 !important;
    border: 1px solid #1e1e28 !important;
    border-radius: 10px !important;
    color: #E8E6E0 !important;
}
.stMultiSelect span[data-baseweb="tag"] {
    background: rgba(232,184,75,0.15) !important;
    color: #E8B84B !important;
    border: 1px solid rgba(232,184,75,0.3) !important;
    border-radius: 6px !important;
}
.stSlider > div > div > div > div { background: #E8B84B !important; }

/* Labels */
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stSlider label,
.stMultiSelect label {
    color: #666 !important;
    font-size: 12px !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* Button */
.stButton > button {
    background: #E8B84B !important;
    color: #0C0C0F !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #f0c558 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(232,184,75,0.25) !important;
}

/* Result card */
.result-card {
    background: #0e0e12;
    border: 1px solid #1e1e28;
    border-radius: 16px;
    padding: 40px 32px;
    text-align: center;
}
.result-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 16px;
}
.result-amount {
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    font-weight: 900;
    color: #E8B84B;
    line-height: 1;
    letter-spacing: -1px;
}
.result-range {
    font-size: 13px;
    color: #444;
    margin-top: 8px;
    font-family: 'DM Mono', monospace;
}
.result-divider { border: none; border-top: 1px solid #1a1a22; margin: 24px 0; }
.tier-pill {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 100px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 16px;
    letter-spacing: 0.3px;
}
.tier-blockbuster { background: rgba(232,143,0,0.15); color: #E88F00; border: 1px solid rgba(232,143,0,0.25); }
.tier-hit         { background: rgba(232,184,75,0.12); color: #E8B84B; border: 1px solid rgba(232,184,75,0.2); }
.tier-modest      { background: rgba(144,202,249,0.1); color: #90CAF9; border: 1px solid rgba(144,202,249,0.2); }
.tier-flop        { background: rgba(239,154,154,0.1); color: #EF9A9A; border: 1px solid rgba(239,154,154,0.2); }

/* Metric cards */
[data-testid="stMetricValue"] { color: #E8B84B !important; font-family: 'Playfair Display', serif !important; font-size: 24px !important; }
[data-testid="stMetricLabel"] { color: #444 !important; font-size: 10px !important; font-family: 'DM Mono', monospace !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
[data-testid="metric-container"] {
    background: #0e0e12 !important;
    border: 1px solid #1a1a22 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

.stProgress > div > div > div { background: linear-gradient(90deg, #1a1a22, #E8B84B) !important; border-radius: 4px !important; }
.stProgress > div > div { background: #111116 !important; border-radius: 4px !important; }
.stCaption { color: #444 !important; font-family: 'DM Mono', monospace !important; font-size: 10px !important; letter-spacing: 1px !important; }
.stAlert { background: #111116 !important; border: 1px solid #1e1e28 !important; border-radius: 12px !important; color: #666 !important; }
.stSpinner > div { border-top-color: #E8B84B !important; }
div[data-testid="stHorizontalBlock"] { gap: 16px; }
hr { border-color: #1a1a22 !important; }

/* ── About page ── */
.about-hero {
    padding: 72px 0 56px;
    border-bottom: 1px solid #1a1a22;
    margin-bottom: 56px;
}
.about-hero-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 4px;
    color: #E8B84B;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.about-hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 56px;
    font-weight: 900;
    color: #E8E6E0;
    line-height: 1.05;
    letter-spacing: -1px;
    max-width: 640px;
}
.about-hero-desc {
    font-size: 17px;
    color: #555;
    line-height: 1.9;
    max-width: 580px;
    margin-top: 24px;
}
.model-card {
    border: 1px solid #1a1a22;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 16px;
}
.model-card-header {
    background: #111116;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #1a1a22;
}
.model-card-title { font-size: 14px; font-weight: 600; color: #E8E6E0; }
.model-card-badge {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(232,184,75,0.12);
    color: #E8B84B;
    border: 1px solid rgba(232,184,75,0.2);
}
.model-card-body { padding: 24px; }
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.m-item {
    padding: 20px;
    background: #0e0e12;
    border: 1px solid #1a1a22;
    border-radius: 10px;
    text-align: center;
}
.m-val {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 700;
    color: #E8B84B;
}
.m-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #444;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 4px;
}
.feat-importance-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.fi-name { font-size: 13px; color: #888; width: 160px; flex-shrink: 0; font-family: 'DM Mono', monospace; }
.fi-bar-wrap { flex: 1; height: 6px; background: #1a1a22; border-radius: 4px; overflow: hidden; }
.fi-bar { height: 100%; border-radius: 4px; background: #E8B84B; }
.fi-val { font-family: 'DM Mono', monospace; font-size: 11px; color: #444; width: 40px; text-align: right; flex-shrink: 0; }

.tech-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 32px;
}
.tech-pill {
    padding: 16px;
    background: #0e0e12;
    border: 1px solid #1a1a22;
    border-radius: 10px;
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #555;
}
.tech-pill-name { font-size: 14px; font-weight: 500; color: #E8E6E0; margin-bottom: 4px; }

.data-stat-row {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 16px;
    margin-top: 32px;
}
.ds-card {
    padding: 28px 20px;
    background: #0e0e12;
    border: 1px solid #1a1a22;
    border-radius: 12px;
    text-align: center;
}
.ds-num { font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 700; color: #E8B84B; }
.ds-lbl { font-size: 12px; color: #444; margin-top: 4px; font-family: 'DM Mono', monospace; letter-spacing: 1px; text-transform: uppercase; }

.pipeline-steps {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: #1a1a22;
    border: 1px solid #1a1a22;
    border-radius: 12px;
    overflow: hidden;
    margin-top: 28px;
}
.pl-step { background: #0C0C0F; padding: 24px 16px; text-align: center; }
.pl-icon { font-size: 20px; margin-bottom: 8px; }
.pl-name { font-size: 12px; font-weight: 600; color: #E8E6E0; margin-bottom: 4px; }
.pl-desc { font-size: 11px; color: #444; line-height: 1.5; }

.footer {
    border-top: 1px solid #1a1a22;
    padding: 40px 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 80px;
}
.footer-logo { font-family: 'Playfair Display', serif; font-size: 18px; font-weight: 900; color: #E8E6E0; }
.footer-logo span { color: #E8B84B; }
.footer-text { font-size: 12px; color: #333; font-family: 'DM Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model           = joblib.load('model_artifacts/model.joblib')
    le              = joblib.load('model_artifacts/language_encoder.joblib')
    mlb             = joblib.load('model_artifacts/genre_mlb.joblib')
    actor_mean_rev  = joblib.load('model_artifacts/actor_mean_rev.joblib')
    director_mean   = joblib.load('model_artifacts/director_mean.joblib')
    global_mean_rev = joblib.load('model_artifacts/global_mean_rev.joblib')
    feature_cols    = joblib.load('model_artifacts/feature_columns.joblib')
    year_budget_map = joblib.load('model_artifacts/year_budget_map.joblib')
    return model, le, mlb, actor_mean_rev, director_mean, global_mean_rev, feature_cols, year_budget_map

model, le, mlb, actor_mean_rev, director_mean, global_mean_rev, feature_cols, year_budget_map = load_artifacts()

GENRE_LIST = sorted(mlb.classes_.tolist())
STUDIO_LIST = [
    'Warner Bros. Pictures', 'Universal Pictures', 'Columbia Pictures',
    'Walt Disney Pictures', 'Marvel Studios', 'Paramount Pictures',
    'New Line Cinema', 'Lionsgate'
]
MONTHS = ['January','February','March','April','May','June',
          'July','August','September','October','November','December']

# ── Helpers ───────────────────────────────────────────────────────
def fmt(n):
    if n >= 1e9: return f"${n/1e9:.2f}B"
    if n >= 1e6: return f"${n/1e6:.1f}M"
    return f"${n/1e3:.0f}K"

def get_tier(revenue):
    if revenue < 20e6:  return "Box Office Flop",   "tier-flop",        "🔴"
    if revenue < 100e6: return "Modest Performer",  "tier-modest",      "🔵"
    if revenue < 400e6: return "Solid Hit",         "tier-hit",         "🟡"
    return                     "Blockbuster",       "tier-blockbuster", "🟠"

def predict(budget, year, month, runtime, language, director,
            genres, cast_list, studios, vote_avg, vote_count, popularity):
    feat = {}
    if language in le.classes_:
        feat['original_language'] = le.transform([language])[0]
    else:
        feat['original_language'] = 0
    feat['year']  = year
    feat['month'] = month
    genre_vec = mlb.transform([genres])[0]
    for g, v in zip(mlb.classes_, genre_vec):
        feat[g] = int(v)
    feat['runtime_cat'] = 0 if runtime <= 90 else (1 if runtime <= 120 else 2)
    feat['budget_log']          = np.log1p(budget)
    feat['budget_x_runtime']    = feat['budget_log'] * feat['runtime_cat']
    mean_budget_year            = year_budget_map.get(year, np.mean(list(year_budget_map.values())))
    feat['budget_relative']     = budget / (mean_budget_year + 1)
    feat['budget_relative_log'] = np.log1p(feat['budget_relative'])
    feat['vote_average']   = vote_avg
    feat['vote_count_log'] = np.log1p(vote_count)
    feat['popularity_log'] = np.log1p(popularity)
    feat['vote_x_avg']     = vote_count * vote_avg
    feat['cast_mean_rev'] = float(np.mean(
        [actor_mean_rev.get(a, global_mean_rev) for a in cast_list]
    )) if cast_list else global_mean_rev
    feat['director_mean_rev'] = float(director_mean.get(director, global_mean_rev))
    for studio in STUDIO_LIST:
        feat[f'studio_{studio}'] = 1 if studio in studios else 0
    feat['genre_count'] = len(genres)
    row = pd.DataFrame([feat])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols]
    log_pred = model.predict(row)[0]
    return float(np.expm1(log_pred))

# ── Session state for navigation ─────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ── Navbar ────────────────────────────────────────────────────────
active = st.session_state.page
st.markdown(f"""
<div class="navbar">
    <div class="nav-logo">Cine<span>Predict</span></div>
    <div class="nav-links" id="nav-links"></div>
</div>
""", unsafe_allow_html=True)

col_n1, col_n2, col_n3, col_n4 = st.columns([4, 1, 1, 1])
with col_n2:
    if st.button("Home", key="nav_home"):
        st.session_state.page = 'home'
        st.rerun()
with col_n3:
    if st.button("Predict", key="nav_predict"):
        st.session_state.page = 'predict'
        st.rerun()
with col_n4:
    if st.button("About", key="nav_about"):
        st.session_state.page = 'about'
        st.rerun()

# ════════════════════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════════════════════
if st.session_state.page == 'home':
    st.markdown('<div class="page">', unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">AI-Powered Box Office Intelligence</div>
        <div class="hero-title">
            Forecast Your Film's
            <span class="hero-title-accent">Box Office.</span>
        </div>
        <div class="hero-sub">
            Enter your movie's details and our XGBoost model — trained on 4,600+ films — 
            predicts revenue before a single frame is shot.
        </div>
        <div class="hero-stats">
            <div class="stat-item">
                <div class="stat-num">4,604</div>
                <div class="stat-label">Training Films</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">0.73</div>
                <div class="stat-label">R² Score</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">41</div>
                <div class="stat-label">Features</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">19</div>
                <div class="stat-label">Genres</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features grid
    st.markdown("""
    <div class="features-grid">
        <div class="feat-card">
            <span class="feat-icon">🎯</span>
            <div class="feat-title">Revenue Prediction</div>
            <div class="feat-desc">Predict worldwide box office gross with confidence ranges, powered by gradient-boosted decision trees trained on decades of cinema data.</div>
        </div>
        <div class="feat-card">
            <span class="feat-icon">💺</span>
            <div class="feat-title">Tier Classification</div>
            <div class="feat-desc">Know instantly whether your project is tracking towards a Blockbuster, Solid Hit, Modest Performer, or needs a rethink.</div>
        </div>
        <div class="feat-card">
            <span class="feat-icon">📊</span>
            <div class="feat-title">ROI Estimation</div>
            <div class="feat-desc">Compare predicted revenue against your production budget to get an estimated return on investment before greenlit.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div style="margin-bottom:16px">
        <div class="section-eyebrow">How it works</div>
        <div class="section-title">Four steps to your forecast</div>
    </div>
    <div class="steps-grid">
        <div class="step-card">
            <div class="step-num">01</div>
            <div class="step-title">Enter Film Details</div>
            <div class="step-desc">Budget, runtime, release month, language and director.</div>
        </div>
        <div class="step-card">
            <div class="step-num">02</div>
            <div class="step-title">Add Cast & Genre</div>
            <div class="step-desc">Actor star power and genre mix are among the top predictors.</div>
        </div>
        <div class="step-card">
            <div class="step-num">03</div>
            <div class="step-title">Set Audience Signals</div>
            <div class="step-desc">Vote average, popularity score and expected vote count.</div>
        </div>
        <div class="step-card">
            <div class="step-num">04</div>
            <div class="step-title">Get Your Forecast</div>
            <div class="step-desc">Instant revenue prediction with low/high range and ROI estimate.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA
    st.markdown("""
    <div class="cta-banner" style="margin-top:56px">
        <div class="cta-title">Ready to predict?</div>
        <div class="cta-sub">Takes less than 2 minutes. No sign-up required.</div>
    </div>
    """, unsafe_allow_html=True)

    col_cta1, col_cta2, col_cta3 = st.columns([2, 1, 2])
    with col_cta2:
        if st.button("Start Predicting →", key="cta_btn"):
            st.session_state.page = 'predict'
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PREDICTION PAGE
# ════════════════════════════════════════════════════════════════
elif st.session_state.page == 'predict':
    st.markdown('<div class="page">', unsafe_allow_html=True)

    st.markdown("""
    <div class="pred-header">
        <div class="pred-title">Revenue Predictor</div>
        <div class="pred-sub">Fill in the details below — all fields improve accuracy</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_gap, col_result = st.columns([1.3, 0.08, 1])

    with col_form:
        st.markdown('<div class="form-section-label">Core Details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            budget = st.number_input("Production Budget (USD)", min_value=0, value=25000000, step=1000000)
            year   = st.number_input("Release Year", min_value=1960, max_value=2030, value=2024)
            language = st.selectbox("Language", ['en','fr','es','de','ja','ko','zh','hi','it','pt'],
                format_func=lambda x: {'en':'English','fr':'French','es':'Spanish',
                'de':'German','ja':'Japanese','ko':'Korean','zh':'Chinese',
                'hi':'Hindi','it':'Italian','pt':'Portuguese'}.get(x, x))
        with c2:
            director = st.text_input("Director", placeholder="e.g. Christopher Nolan")
            month    = st.selectbox("Release Month", range(1,13), format_func=lambda x: MONTHS[x-1], index=5)
            runtime  = st.number_input("Runtime (minutes)", min_value=60, max_value=300, value=110)

        st.markdown('<div class="form-section-label">Cast</div>', unsafe_allow_html=True)
        cast_input = st.text_input("Actors (comma separated)", placeholder="e.g. Tom Hanks, Scarlett Johansson")
        cast_list  = [a.strip() for a in cast_input.split(',') if a.strip()] if cast_input else []

        st.markdown('<div class="form-section-label">Genres</div>', unsafe_allow_html=True)
        genres = st.multiselect("Select Genres", GENRE_LIST, default=['Drama'])

        st.markdown('<div class="form-section-label">Production Studios</div>', unsafe_allow_html=True)
        studios = st.multiselect("Select Studios", STUDIO_LIST)

        st.markdown('<div class="form-section-label">Audience Signals</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            vote_avg   = st.slider("Vote Average", 1.0, 10.0, 7.0, 0.1)
            popularity = st.number_input("Popularity Score", min_value=0, value=50)
        with c4:
            vote_count = st.number_input("Est. Vote Count", min_value=0, value=3000)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Predict Revenue", key="predict_main")

    with col_result:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown('<div class="form-section-label">Your Forecast</div>', unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Analysing..."):
                revenue = predict(
                    budget, year, month, runtime, language, director,
                    genres, cast_list, studios, vote_avg, vote_count, popularity
                )

            low  = revenue * 0.75
            high = revenue * 1.25
            tier_label, tier_cls, tier_icon = get_tier(revenue)
            roi  = ((revenue - budget) / budget * 100) if budget > 0 else 0

            st.markdown(f"""
            <div class="result-card">
                <div class="result-eyebrow">Predicted Worldwide Revenue</div>
                <div class="result-amount">{fmt(revenue)}</div>
                <div class="result-range">{fmt(low)} — {fmt(high)}</div>
                <hr class="result-divider">
                <span class="tier-pill {tier_cls}">{tier_icon} {tier_label}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.caption("Revenue Scale  ·  $0 → $1.5B+")
            st.progress(min(revenue / 1.5e9, 1.0))

            st.markdown("<br>", unsafe_allow_html=True)

            s1, s2 = st.columns(2)
            with s1:
                st.metric("Predicted", fmt(revenue))
                st.metric("Low Estimate", fmt(low))
            with s2:
                st.metric("ROI Estimate", f"{roi:.0f}%")
                st.metric("High Estimate", fmt(high))

        else:
            st.markdown("""
            <div style="border:1px solid #1a1a22;border-radius:12px;padding:48px 32px;text-align:center">
                <div style="font-size:32px;margin-bottom:16px">🎬</div>
                <div style="font-size:14px;color:#444;line-height:1.8;font-family:'DM Mono',monospace">
                    Complete the form<br>and click Predict
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ════════════════════════════════════════════════════════════════
elif st.session_state.page == 'about':
    st.markdown('<div class="page">', unsafe_allow_html=True)

    st.markdown("""
    <div class="about-hero">
        <div class="about-hero-label">About the Model</div>
        <div class="about-hero-title">What's under the hood of CinePredict?</div>
        <div class="about-hero-desc">
            CinePredict uses a gradient-boosted XGBoost model trained on over 4,600 real films. 
            Here's a transparent look at the data, features, performance, and engineering decisions 
            that power the predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset stats
    st.markdown('<div class="section-eyebrow">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:28px;margin-bottom:8px">Training data at a glance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="data-stat-row">
        <div class="ds-card"><div class="ds-num">4,604</div><div class="ds-lbl">Total Films</div></div>
        <div class="ds-card"><div class="ds-num">66</div><div class="ds-lbl">Raw Features</div></div>
        <div class="ds-card"><div class="ds-num">41</div><div class="ds-lbl">Model Features</div></div>
        <div class="ds-card"><div class="ds-num">34</div><div class="ds-lbl">Languages</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model performance
    st.markdown('<div class="section-eyebrow" style="margin-top:32px">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:28px;margin-bottom:20px">XGBoost vs Random Forest</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="model-card">
        <div class="model-card-header">
            <div class="model-card-title">XGBoost Regressor</div>
            <div class="model-card-badge">Final Model</div>
        </div>
        <div class="model-card-body">
            <div class="metric-row">
                <div class="m-item"><div class="m-val">0.73</div><div class="m-lbl">R² Score</div></div>
                <div class="m-item"><div class="m-val">$56M</div><div class="m-lbl">MAE</div></div>
                <div class="m-item"><div class="m-val">$119M</div><div class="m-lbl">RMSE</div></div>
            </div>
            <div style="font-size:13px;color:#444;line-height:1.8">
                Trained with early stopping at 213 rounds (validation RMSE: 0.805). 
                XGBoost outperformed Random Forest (R² 0.66) due to better handling of 
                heterogeneous tabular features, especially target-encoded cast/director signals.
            </div>
        </div>
    </div>

    <div class="model-card">
        <div class="model-card-header">
            <div class="model-card-title">Random Forest Regressor</div>
            <div class="model-card-badge" style="background:rgba(100,100,100,0.1);color:#555;border-color:#1a1a22">Baseline</div>
        </div>
        <div class="model-card-body">
            <div class="metric-row">
                <div class="m-item"><div class="m-val" style="color:#555">0.66</div><div class="m-lbl">R² Score</div></div>
                <div class="m-item"><div class="m-val" style="color:#555">$56M</div><div class="m-lbl">MAE</div></div>
                <div class="m-item"><div class="m-val" style="color:#555">—</div><div class="m-lbl">RMSE</div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="section-eyebrow" style="margin-top:40px">Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:28px;margin-bottom:24px">Top 10 predictors</div>', unsafe_allow_html=True)

    importances = [
        ("budget_log",       43.0),
        ("vote_count_log",   22.0),
        ("vote_average",      6.4),
        ("year",              6.0),
        ("vote_x_avg",        5.1),
        ("popularity_log",    4.9),
        ("budget_x_runtime",  2.8),
        ("month",             2.4),
        ("Action",            0.73),
        ("Comedy",            0.62),
    ]
    max_val = importances[0][1]
    rows_html = ""
    for name, val in importances:
        pct = int(val / max_val * 100)
        rows_html += f"""
        <div class="feat-importance-row">
            <div class="fi-name">{name}</div>
            <div class="fi-bar-wrap"><div class="fi-bar" style="width:{pct}%"></div></div>
            <div class="fi-val">{val:.1f}%</div>
        </div>"""
    st.markdown(f'<div style="padding:24px;background:#0e0e12;border:1px solid #1a1a22;border-radius:14px">{rows_html}</div>', unsafe_allow_html=True)

    # Pipeline
    st.markdown('<div class="section-eyebrow" style="margin-top:48px">Engineering</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:28px;margin-bottom:8px">Data pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-steps">
        <div class="pl-step">
            <div class="pl-icon">📥</div>
            <div class="pl-name">Ingest</div>
            <div class="pl-desc">Raw CSV, 4,604 films filtered for budget > 0 and revenue > 0</div>
        </div>
        <div class="pl-step">
            <div class="pl-icon">🔧</div>
            <div class="pl-name">Engineer</div>
            <div class="pl-desc">Log transforms, budget-relative, genre binarization, runtime categories</div>
        </div>
        <div class="pl-step">
            <div class="pl-icon">🎭</div>
            <div class="pl-name">Encode</div>
            <div class="pl-desc">Target-encode cast & director with mean log-revenue (3+ films threshold)</div>
        </div>
        <div class="pl-step">
            <div class="pl-icon">🤖</div>
            <div class="pl-name">Train</div>
            <div class="pl-desc">XGBoost with early stopping, 80/20 train-val split</div>
        </div>
        <div class="pl-step">
            <div class="pl-icon">💾</div>
            <div class="pl-name">Persist</div>
            <div class="pl-desc">Artifacts saved via joblib: model, encoders, mean maps, feature list</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack
    st.markdown('<div class="section-eyebrow" style="margin-top:48px">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tech-grid">
        <div class="tech-pill"><div class="tech-pill-name">XGBoost</div>Gradient Boosting</div>
        <div class="tech-pill"><div class="tech-pill-name">scikit-learn</div>Preprocessing & Metrics</div>
        <div class="tech-pill"><div class="tech-pill-name">pandas / numpy</div>Data Engineering</div>
        <div class="tech-pill"><div class="tech-pill-name">Streamlit</div>Web Interface</div>
        <div class="tech-pill"><div class="tech-pill-name">joblib</div>Model Persistence</div>
        <div class="tech-pill"><div class="tech-pill-name">Python 3.13</div>Runtime</div>
        <div class="tech-pill"><div class="tech-pill-name">LabelEncoder</div>Language Encoding</div>
        <div class="tech-pill"><div class="tech-pill-name">MLB Binarizer</div>Genre Encoding</div>
    </div>
    """, unsafe_allow_html=True)

    # Limitations
    st.markdown('<div class="section-eyebrow" style="margin-top:48px">Limitations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:28px;margin-bottom:16px">What to keep in mind</div>', unsafe_allow_html=True)
    st.info("""
**R² of 0.73** means the model explains ~73% of variance in log-revenue — strong for tabular film data, but a 27% gap remains unexplained by the available features.

**MAE of $56M** reflects the high variance in box office outcomes: a film can dramatically over or underperform based on marketing, competition, and cultural timing — none of which are in the dataset.

**Vote signals are post-release** — vote_average, vote_count, and popularity are the two strongest predictors but require audience data that may not be available at greenlight. Use them as sensitivity levers or set them to market comparables.

**Target encoding** of cast and director uses historical mean revenue. New or unknown talent defaults to the global mean — predictions for debut directors will be conservative.
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-logo">Cine<span>Predict</span></div>
    <div class="footer-text">XGBoost · R² 0.73 · 4,604 Films · 41 Features</div>
</div>
""", unsafe_allow_html=True)