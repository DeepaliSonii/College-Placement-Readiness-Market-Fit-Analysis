import streamlit as st

st.set_page_config(
    page_title="College Placement Readiness",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#040810;--surface:#080f1e;--surface-2:#0c1526;--border:rgba(99,130,255,0.12);--text:#e8eeff;--muted:#6b7a9e;--accent:#6382ff;--accent-2:#a78bfa;--green:#34d399;--amber:#fbbf24;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:var(--bg);font-family:'DM Sans',sans-serif;color:var(--text);}
.block-container{max-width:1100px;padding:2rem 2rem 4rem;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#060b1a 0%,#040810 100%) !important;border-right:1px solid rgba(99,130,255,0.10) !important;min-width:220px !important;max-width:220px !important;}
section[data-testid="stSidebar"] .block-container{padding:0 !important;}
.sb-brand{padding:1.4rem 1.2rem 1.2rem;border-bottom:1px solid rgba(99,130,255,0.10);margin-bottom:0.5rem;}
.sb-logo{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#f0f4ff;letter-spacing:-0.5px;}
.sb-logo span{color:var(--accent);}
.sb-tagline{font-size:0.68rem;color:var(--muted);margin-top:3px;}
.sb-section-label{font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:rgba(107,122,158,0.55);padding:0.8rem 1.2rem 0.3rem;}
.sb-nav{display:flex;align-items:center;gap:10px;padding:0.55rem 1rem;margin:1px 0.5rem;border-radius:9px;font-size:0.84rem;color:var(--muted);border:1px solid transparent;}
.sb-nav.active{background:linear-gradient(135deg,rgba(99,130,255,0.18),rgba(167,139,250,0.10));color:#e0e8ff;border-color:rgba(99,130,255,0.28);font-weight:500;}
.sb-icon{font-size:0.9rem;width:18px;text-align:center;flex-shrink:0;}
/* hide the duplicate streamlit nav buttons that appear below custom nav */
section[data-testid="stSidebar"] .stButton>button{
    background:transparent !important;
    border:none !important;
    color:var(--muted) !important;
    text-align:left !important;
    padding:0.55rem 1rem !important;
    margin:1px 0.5rem !important;
    border-radius:9px !important;
    font-size:0.84rem !important;
    font-family:'DM Sans',sans-serif !important;
    font-weight:400 !important;
    letter-spacing:normal !important;
    text-transform:none !important;
    box-shadow:none !important;
    width:calc(100% - 1rem) !important;
}
section[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(99,130,255,0.07) !important;
    color:#c7d2fe !important;
    transform:none !important;
}
.page-header{margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid var(--border);}
.page-title{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;color:#f0f4ff;line-height:1.1;margin:0 0 0.4rem;}
.page-title span{color:var(--accent);}
.page-subtitle{color:var(--muted);font-size:0.95rem;font-weight:300;}
.section-label{font-family:'Syne',sans-serif;font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--muted);margin-bottom:0.8rem;}
[data-testid="stFileUploader"]{background:linear-gradient(135deg,rgba(99,130,255,0.04),rgba(167,139,250,0.04));border:1.5px dashed rgba(99,130,255,0.25) !important;border-radius:14px !important;padding:0.5rem;}
div[data-baseweb="input"]>div{background:var(--surface-2) !important;border:1px solid rgba(99,130,255,0.18) !important;border-radius:10px !important;}
div[data-baseweb="input"] input{color:var(--text) !important;font-family:'DM Sans',sans-serif;}
div[data-baseweb="select"]>div{background:var(--surface-2) !important;border:1px solid rgba(99,130,255,0.18) !important;border-radius:10px !important;}
div[data-baseweb="select"] span,div[data-baseweb="select"] div{color:var(--text) !important;}
[data-testid="stSlider"] [role="slider"]{background:var(--accent) !important;border:2px solid #fff !important;}
.stButton>button{width:100%;background:linear-gradient(135deg,#4b6eff,#7c3aed);color:white !important;border:none !important;border-radius:12px;padding:0.85rem 1.5rem;font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;transition:all 0.2s;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(99,130,255,0.35);}
label,.stSlider label,.stNumberInput label{color:var(--muted) !important;font-family:'DM Sans',sans-serif !important;font-size:0.82rem !important;text-transform:uppercase !important;}
.stAlert{border-radius:10px !important;}
hr{border-color:var(--border) !important;margin:1.5rem 0 !important;}
.badge-row{display:flex;gap:10px;margin:1rem 0 0;flex-wrap:wrap;}
.badge{background:rgba(99,130,255,0.08);border:1px solid rgba(99,130,255,0.18);border-radius:8px;padding:5px 12px;font-size:0.75rem;color:var(--accent);}
.steps{display:flex;gap:8px;align-items:center;margin-bottom:2rem;flex-wrap:wrap;}
.step{display:flex;align-items:center;gap:8px;font-size:0.8rem;color:var(--muted);}
.step-num{width:24px;height:24px;border-radius:50%;border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:600;font-family:'Syne',sans-serif;}
.step.active .step-num{background:var(--accent);border-color:var(--accent);color:white;}
.step-divider{flex:1;height:1px;background:var(--border);max-width:40px;}
.section-heading{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:#f0f4ff;margin:2rem 0 1rem;display:flex;align-items:center;gap:10px;}
.section-heading::after{content:'';flex:1;height:1px;background:var(--border);}
.how-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:1.5rem;}
.how-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.2rem 1.3rem;position:relative;overflow:hidden;}
.how-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.how-card:nth-child(1)::before{background:linear-gradient(90deg,#6382ff,transparent);}
.how-card:nth-child(2)::before{background:linear-gradient(90deg,#a78bfa,transparent);}
.how-card:nth-child(3)::before{background:linear-gradient(90deg,#34d399,transparent);}
.how-num{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:rgba(99,130,255,0.15);line-height:1;margin-bottom:0.5rem;}
.how-title{font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;color:#e8eeff;margin-bottom:0.4rem;}
.how-desc{font-size:0.8rem;color:var(--muted);line-height:1.6;}
.score-guide{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1.5rem;}
.sg-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:0.9rem 1rem;text-align:center;}
.sg-range{font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;margin-bottom:3px;}
.sg-label{font-size:0.72rem;color:var(--muted);}
.role-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.5rem;}
.role-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.1rem;transition:border-color 0.2s;}
.role-card:hover{border-color:rgba(99,130,255,0.35);background:rgba(99,130,255,0.04);}
.role-emoji{font-size:1.4rem;margin-bottom:0.4rem;}
.role-name{font-family:'Syne',sans-serif;font-size:0.82rem;font-weight:700;color:#e8eeff;margin-bottom:0.3rem;}
.role-stat{font-size:0.73rem;color:var(--muted);margin-bottom:2px;}
.role-stat span{color:var(--green);font-weight:600;}
.tips-box{background:linear-gradient(135deg,rgba(99,130,255,0.05),rgba(167,139,250,0.03));border:1px solid rgba(99,130,255,0.15);border-radius:14px;padding:1.2rem 1.4rem;}
.tips-title{font-family:'Syne',sans-serif;font-size:0.78rem;font-weight:700;color:var(--accent);margin-bottom:0.8rem;letter-spacing:0.06em;text-transform:uppercase;}
.tip-row{display:flex;align-items:flex-start;gap:10px;margin-bottom:0.5rem;font-size:0.82rem;color:#a5b4fc;line-height:1.55;}
.tip-dot{width:5px;height:5px;border-radius:50%;background:var(--accent);margin-top:7px;flex-shrink:0;}
</style>
"""
st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""

<div class="sb-section-label">Navigation</div>
<div class="sb-nav active"><span class="sb-icon">🏠</span> Home</div>
""", unsafe_allow_html=True)
    if st.button("📊  Readiness Dashboard", key="nav_r"):  st.switch_page("pages/readiness.py")
    if st.button("🔍  Skill Gap Analysis",   key="nav_sg"): st.switch_page("pages/skill_gap.py")
    if st.button("✏️   Quick Test",           key="nav_qz"): st.switch_page("pages/quiz.py")

# ── PAGE CONTENT ──
st.markdown("""
<div class="page-header">
    <div class="page-title">College <span>Placement Readiness</span><br>And Market Fit Analysis</div>
    <div class="page-subtitle">Upload your resume and enter academic details to receive your personalized placement analysis.</div>
    <div class="badge-row">
        <div class="badge">ML-Powered Prediction</div>
        <div class="badge">Skill Gap Analysis</div>
        <div class="badge">Career Roadmap</div>
        <div class="badge">Quick Test</div>
    </div>
</div>
<div class="steps">
    <div class="step active"><div class="step-num">1</div><span>Upload &amp; Details</span></div>
    <div class="step-divider"></div>
    <div class="step"><div class="step-num">2</div><span>Readiness Score</span></div>
    <div class="step-divider"></div>
    <div class="step"><div class="step-num">3</div><span>Skill Gap Report</span></div>
    <div class="step-divider"></div>
    <div class="step"><div class="step-num">4</div><span>Test</span></div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.1, 1], gap="large")
with left:
    st.markdown('<div class="section-label">Resume Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["pdf", "docx"], label_visibility="collapsed")
with right:
    st.markdown('<div class="section-label">Academic Profile</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1: cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1, format="%.2f")
    with a2: backlogs = st.number_input("Active Backlogs", min_value=0, max_value=10, value=0)

st.markdown("---")
st.markdown('<div class="section-label">Proficiency Assessment</div>', unsafe_allow_html=True)
s1, s2 = st.columns(2, gap="large")
with s1: communication = st.slider("Communication Skills", 1, 10, 5)
with s2: aptitude = st.slider("Aptitude Score", 0, 100, 60)

st.markdown("---")
st.markdown('<div class="section-label">Career Target</div>', unsafe_allow_html=True)
target_role = st.selectbox("Role", ["Data Scientist", "Software Engineer", "ML Engineer", "Frontend Developer"], label_visibility="collapsed")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
analyze = st.button("Analyze My Profile →", use_container_width=True)

if analyze:
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.cgpa = cgpa
        st.session_state.communication = communication
        st.session_state.aptitude = aptitude
        st.session_state.backlogs = backlogs
        st.session_state.target_role = target_role
        st.switch_page("pages/readiness.py")
    else:
        st.error("Please upload your resume before continuing.")

st.markdown('<div class="section-heading">How It Works</div>', unsafe_allow_html=True)
st.markdown("""
<div class="how-grid">
<div class="how-card"><div class="how-num">01</div><div class="how-title">Resume Parsing</div><div class="how-desc">Our system scans your resume to extract skills, projects, internships, and certifications automatically.</div></div>
<div class="how-card"><div class="how-num">02</div><div class="how-title">ML-Based Prediction</div><div class="how-desc">An XGBoost model trained on real placement data calculates your readiness score and predicts placement probability.</div></div>
<div class="how-card"><div class="how-num">03</div><div class="how-title">Skill Gap + Quick Test</div><div class="how-desc">Compare your profile against benchmarks, get a roadmap, and take a quick aptitude test.</div></div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-heading">Readiness Score Guide</div>', unsafe_allow_html=True)
st.markdown("""
<div class="score-guide">
<div class="sg-card"><div class="sg-range" style="color:#f87171">0–40</div><div class="sg-label">Not Ready</div></div>
<div class="sg-card"><div class="sg-range" style="color:#fbbf24">41–60</div><div class="sg-label">Moderately Ready</div></div>
<div class="sg-card"><div class="sg-range" style="color:#60a5fa">61–80</div><div class="sg-label">Mostly Ready</div></div>
<div class="sg-card"><div class="sg-range" style="color:#34d399">81–100</div><div class="sg-label">Placement Ready</div></div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-heading">Role Benchmarks</div>', unsafe_allow_html=True)
st.markdown("""
<div class="role-grid">
<div class="role-card"><div class="role-emoji">🧠</div><div class="role-name">Data Scientist</div><div class="role-stat">Min CGPA: <span>7.5+</span></div><div class="role-stat">Skills: <span>10–12</span></div><div class="role-stat">Key: Python, ML, SQL</div></div>
<div class="role-card"><div class="role-emoji">💻</div><div class="role-name">Software Engineer</div><div class="role-stat">Min CGPA: <span>7.0+</span></div><div class="role-stat">Skills: <span>8–10</span></div><div class="role-stat">Key: DSA, OOP, Git</div></div>
<div class="role-card"><div class="role-emoji">🤖</div><div class="role-name">ML Engineer</div><div class="role-stat">Min CGPA: <span>8.0+</span></div><div class="role-stat">Skills: <span>10–14</span></div><div class="role-stat">Key: PyTorch, MLOps</div></div>
<div class="role-card"><div class="role-emoji">🎨</div><div class="role-name">Frontend Developer</div><div class="role-stat">Min CGPA: <span>6.5+</span></div><div class="role-stat">Skills: <span>6–9</span></div><div class="role-stat">Key: React, CSS, JS</div></div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-heading">Tips to Maximize Your Score</div>', unsafe_allow_html=True)
st.markdown("""
<div class="tips-box">
<div class="tips-title">💡 Before You Submit</div>
<div class="tip-row"><div class="tip-dot"></div>List all technical skills, tools, and languages clearly — the AI parser reads them and counts toward your coding score.</div>
<div class="tip-row"><div class="tip-dot"></div>Include every project with its tech stack. Personal, academic, or hackathon projects all count.</div>
<div class="tip-row"><div class="tip-dot"></div>Internship experience is heavily weighted — add any part-time, remote, or virtual internships.</div>
<div class="tip-row"><div class="tip-dot"></div>Online certifications from Coursera, Udemy, NPTEL, or LinkedIn Learning are recognized.</div>
<div class="tip-row"><div class="tip-dot"></div>After analysis, use the AI Career Coach to ask specific questions and take the Quick Test to benchmark yourself.</div>
</div>""", unsafe_allow_html=True)