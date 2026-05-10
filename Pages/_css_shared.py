"""
Shared CSS + sidebar renderer for PlaceIQ.
Import this module and call inject_css() + render_sidebar(active) on every page.
"""
import streamlit as st

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:#040810; --surface:#080f1e; --surface-2:#0c1526;
    --border:rgba(99,130,255,0.12); --text:#e8eeff;
    --muted:#6b7a9e; --accent:#6382ff; --accent-2:#a78bfa;
    --green:#34d399; --amber:#fbbf24; --red:#f87171;
    --sidebar-w:230px;
}

#MainMenu,footer,header{visibility:hidden;}
.stApp{background:var(--bg);font-family:'DM Sans',sans-serif;color:var(--text);}
.block-container{max-width:1180px;padding:2rem 2rem 4rem;}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#060b1a 0%,#040810 100%) !important;
    border-right:1px solid rgba(99,130,255,0.10) !important;
    min-width:220px !important; max-width:220px !important;
}
section[data-testid="stSidebar"] .block-container{padding:1.5rem 0 2rem !important;}

.sb-brand{
    padding:0 1.2rem 1.4rem;
    border-bottom:1px solid rgba(99,130,255,0.10);
    margin-bottom:1.2rem;
}
.sb-logo{
    font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800;
    color:#f0f4ff; letter-spacing:-0.5px; line-height:1;
}
.sb-logo span{color:var(--accent);}
.sb-tagline{font-size:0.68rem;color:var(--muted);margin-top:3px;letter-spacing:0.04em;}

.sb-section-label{
    font-size:0.62rem; letter-spacing:0.14em; text-transform:uppercase;
    color:rgba(107,122,158,0.6); padding:0 1.2rem; margin-bottom:0.4rem;
}
.sb-nav-item{
    display:flex; align-items:center; gap:10px;
    padding:0.58rem 1.2rem; margin:1px 0.5rem;
    border-radius:9px; cursor:pointer;
    font-size:0.85rem; color:var(--muted);
    border:1px solid transparent;
    transition:all 0.18s; text-decoration:none;
}
.sb-nav-item:hover{
    background:rgba(99,130,255,0.07);
    color:#c7d2fe; border-color:rgba(99,130,255,0.15);
}
.sb-nav-item.active{
    background:linear-gradient(135deg,rgba(99,130,255,0.18),rgba(167,139,250,0.10));
    color:#e0e8ff !important; border-color:rgba(99,130,255,0.30);
    font-weight:500;
}
.sb-icon{font-size:0.95rem;width:18px;text-align:center;flex-shrink:0;}
.sb-divider{height:1px;background:rgba(99,130,255,0.08);margin:0.8rem 1rem;}
.sb-footer{
    padding:0.9rem 1.2rem; margin-top:0.5rem;
    border-top:1px solid rgba(99,130,255,0.08);
    font-size:0.72rem; color:rgba(107,122,158,0.5); line-height:1.55;
}

/* ── COMMON COMPONENTS ── */
.page-header{margin-bottom:2rem;}
.header-eyebrow{font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--accent);margin-bottom:0.4rem;}
.page-title{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#f0f4ff;margin:0 0 0.3rem;}
.page-subtitle{color:var(--muted);font-size:0.85rem;font-weight:300;}

.section-heading{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f4ff;margin:1.8rem 0 1rem;display:flex;align-items:center;gap:10px;}
.section-heading::after{content:'';flex:1;height:1px;background:var(--border);}

.stAlert{border-radius:10px !important;}
.stButton>button{background:linear-gradient(135deg,#4b6eff,#7c3aed);color:white !important;border:none !important;border-radius:12px;padding:0.8rem 1.5rem;font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;transition:all 0.2s;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(99,130,255,0.35);}
div[data-baseweb="input"] > div{background:var(--surface-2) !important;border:1px solid rgba(99,130,255,0.18) !important;border-radius:10px !important;}
div[data-baseweb="input"] input{color:var(--text) !important;font-family:'DM Sans',sans-serif;}
div[data-baseweb="select"] > div{background:var(--surface-2) !important;border:1px solid rgba(99,130,255,0.18) !important;border-radius:10px !important;}
div[data-baseweb="select"] span,div[data-baseweb="select"] div{color:var(--text) !important;font-family:'DM Sans',sans-serif;}
[data-testid="stFileUploader"]{background:linear-gradient(135deg,rgba(99,130,255,0.04),rgba(167,139,250,0.04));border:1.5px dashed rgba(99,130,255,0.25) !important;border-radius:14px !important;padding:0.5rem;transition:border-color 0.2s;}
[data-testid="stSlider"] [role="slider"]{background:var(--accent) !important;border:2px solid #fff !important;box-shadow:0 0 10px rgba(99,130,255,0.5) !important;}
label,.stSlider label,.stNumberInput label,.stFileUploader label{color:var(--muted) !important;font-family:'DM Sans',sans-serif !important;font-size:0.82rem !important;}
hr{border-color:var(--border) !important;margin:1.5rem 0 !important;}
</style>
"""

def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# page_key: "home" | "readiness" | "skill_gap" | "chatbot" | "quiz"
NAV_ITEMS = [
    ("home",      "🏠", "Home",            "app.py"),
    ("readiness", "📊", "Readiness",       "pages/readiness.py"),
    ("skill_gap", "🔍", "Skill Gap",       "pages/skill_gap.py"),
    ("quiz",      "✏️",  "Quick Test",      "pages/quiz.py"),
]

def render_sidebar(active: str):
    with st.sidebar:
        st.markdown("""
<div class="sb-section-label">Navigation</div>
""", unsafe_allow_html=True)

        pages = {
            "home":      "app",
            "readiness": "pages/readiness",
            "skill_gap": "pages/skill_gap",
         
            "quiz":      "pages/quiz",
        }
        labels = {
            "home":      ("🏠", "Home"),
            "readiness": ("📊", "Readiness Dashboard"),
            "skill_gap": ("🔍", "Skill Gap Analysis"),
            "quiz":      ("✏️", "Quick Test"),
        }
        for key, (icon, label) in labels.items():
            cls = "sb-nav-item active" if key == active else "sb-nav-item"
            st.markdown(f'<div class="{cls}"><span class="sb-icon">{icon}</span>{label}</div>', unsafe_allow_html=True)
            # Streamlit button for navigation
            if key != active:
                if st.button(label, key=f"nav_{key}", help=f"Go to {label}"):
                    st.switch_page(f"{pages[key]}.py")

        st.markdown("""

""", unsafe_allow_html=True)