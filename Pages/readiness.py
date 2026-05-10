import streamlit as st
import pandas as pd
import tempfile
import plotly.graph_objects as go
import pickle

st.set_page_config(
    page_title="Readiness Dashboard ·",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

from resume_parser import extract_resume_text, parse_resume_with_ai
from model_training import calculate_readiness_score, get_readiness_label

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#040810;--surface:#080f1e;--surface-2:#0c1526;--border:rgba(99,130,255,0.12);--text:#e8eeff;--muted:#6b7a9e;--accent:#6382ff;--accent-2:#a78bfa;--green:#34d399;--amber:#fbbf24;--red:#f87171;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:var(--bg);font-family:'DM Sans',sans-serif;color:var(--text);}
.block-container{max-width:1180px;padding:2rem 2rem 4rem;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#060b1a 0%,#040810 100%) !important;border-right:1px solid rgba(99,130,255,0.10) !important;min-width:220px !important;max-width:220px !important;}
section[data-testid="stSidebar"] .block-container{padding:0 !important;}
.sb-brand{padding:1.4rem 1.2rem 1.2rem;border-bottom:1px solid rgba(99,130,255,0.10);margin-bottom:0.5rem;}
.sb-logo{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#f0f4ff;letter-spacing:-0.5px;}
.sb-logo span{color:#6382ff;}
.sb-tagline{font-size:0.68rem;color:#6b7a9e;margin-top:3px;}
.sb-section-label{font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:rgba(107,122,158,0.55);padding:0.8rem 1.2rem 0.3rem;}
.sb-nav{display:flex;align-items:center;gap:10px;padding:0.55rem 1rem;margin:1px 0.5rem;border-radius:9px;font-size:0.84rem;color:#6b7a9e;border:1px solid transparent;}
.sb-nav.active{background:linear-gradient(135deg,rgba(99,130,255,0.18),rgba(167,139,250,0.10));color:#e0e8ff;border-color:rgba(99,130,255,0.28);font-weight:500;}
.sb-icon{font-size:0.9rem;width:18px;text-align:center;flex-shrink:0;}
section[data-testid="stSidebar"] .stButton>button{background:transparent !important;border:none !important;color:#6b7a9e !important;text-align:left !important;padding:0.55rem 1rem !important;margin:1px 0.5rem !important;border-radius:9px !important;font-size:0.84rem !important;font-family:'DM Sans',sans-serif !important;font-weight:400 !important;letter-spacing:normal !important;text-transform:none !important;box-shadow:none !important;width:calc(100% - 1rem) !important;}
section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(99,130,255,0.07) !important;color:#c7d2fe !important;transform:none !important;}
.page-header{margin-bottom:2rem;}
.header-eyebrow{font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--accent);margin-bottom:0.4rem;}
.page-title{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#f0f4ff;margin:0 0 0.3rem;}
.page-subtitle{color:var(--muted);font-size:0.85rem;font-weight:300;}
.stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:1.5rem 0 2rem;}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.4rem 1.5rem;position:relative;overflow:hidden;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.stat-card.green::before{background:linear-gradient(90deg,#34d399,transparent);}
.stat-card.blue::before{background:linear-gradient(90deg,#60a5fa,transparent);}
.stat-card.amber::before{background:linear-gradient(90deg,#fbbf24,transparent);}
.stat-label{font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);margin-bottom:0.6rem;}
.stat-value{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;line-height:1;margin-bottom:0.2rem;}
.stat-card.green .stat-value{color:#34d399;}
.stat-card.blue .stat-value{color:#60a5fa;}
.stat-card.amber .stat-value{color:#fbbf24;}
.stat-note{font-size:0.78rem;color:var(--muted);}
.section-heading{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f4ff;margin:1.8rem 0 1rem;display:flex;align-items:center;gap:10px;}
.section-heading::after{content:'';flex:1;height:1px;background:var(--border);}
.chip-grid{display:flex;flex-wrap:wrap;gap:10px;margin:0.5rem 0 1.5rem;}
.chip{background:rgba(99,130,255,0.07);border:1px solid rgba(99,130,255,0.2);color:#a5b4fc;border-radius:8px;padding:7px 16px;font-size:0.85rem;font-weight:500;}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:1rem 0;}
.metric-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.2rem;text-align:center;}
.metric-num{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:var(--accent);margin-bottom:2px;}
.metric-lbl{font-size:0.75rem;color:var(--muted);letter-spacing:0.08em;text-transform:uppercase;}
.stButton>button{background:linear-gradient(135deg,#4b6eff,#7c3aed);color:white !important;border:none !important;border-radius:12px;padding:0.8rem 1.5rem;font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;transition:all 0.2s;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(99,130,255,0.35);}
.stAlert{border-radius:10px !important;}
.breakdown-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:0.5rem 0 1.5rem;}
.breakdown-item{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.2rem;}
.bi-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
.bi-label{font-size:0.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.07em;}
.bi-val{font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;color:#e8eeff;}
.bi-bar-bg{height:5px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;}
.bi-bar-fill{height:5px;border-radius:99px;}
.bench-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:0.5rem 0 1.5rem;}
.bench-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.2rem;}
.bench-title{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;}
.bench-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;}
.bench-label{font-size:0.78rem;color:#8895b8;}
.bench-you{font-family:'Syne',sans-serif;font-size:0.82rem;font-weight:700;}
.bench-bar-wrap{height:4px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;margin-top:6px;}
.bench-bar{height:4px;border-radius:99px;}
.next-card{background:var(--surface);border:1px solid var(--border);border-left:2px solid var(--accent);border-radius:12px;padding:1rem 1.1rem;margin-bottom:10px;}
.next-tag{font-size:0.67rem;text-transform:uppercase;letter-spacing:0.09em;color:var(--accent);margin-bottom:0.4rem;}
.next-text{font-size:0.82rem;color:#c7d2fe;line-height:1.55;}
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""
<div class="sb-brand">
</div>
<div class="sb-section-label">Navigation</div>
""", unsafe_allow_html=True)
    if st.button("🏠  Home", key="sb_home"): st.switch_page("app.py")
    st.markdown('<div class="sb-nav active"><span class="sb-icon">📊</span> Readiness Dashboard</div>', unsafe_allow_html=True)
    if st.button("🔍  Skill Gap Analysis", key="sb_sg"): st.switch_page("pages/skill_gap.py")
    if st.button("✏️   Quick Test",         key="sb_qz"): st.switch_page("pages/quiz.py")

# ── HEADER ──
st.markdown("""

   
    <div class="page-title">Placement Readiness Dashboard</div>
""", unsafe_allow_html=True)

if "uploaded_file" not in st.session_state:
    st.warning("Please upload your resume first from the Home Page.")
    if st.button("← Go to Home"): st.switch_page("app.py")
    st.stop()

uploaded_file = st.session_state.uploaded_file
cgpa          = st.session_state.cgpa
communication = st.session_state.communication
aptitude      = st.session_state.aptitude
backlogs      = st.session_state.backlogs
target_role   = st.session_state.get("target_role", "Software Engineer")

with open("models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

file_extension = uploaded_file.name.split(".")[-1]
with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
    tmp_file.write(uploaded_file.getbuffer())
    temp_path = tmp_file.name

resume_text = extract_resume_text(temp_path)
parsed_data = parse_resume_with_ai(resume_text)

skills         = parsed_data["skills"]
internships    = parsed_data["internships"]
projects       = parsed_data["projects"]
certifications = parsed_data["certifications"]
st.session_state.skills = skills

student_data = {
    "CGPA": cgpa, "Internships": internships, "Projects": projects,
    "Coding_Skills": min(len(skills), 10), "Communication_Skills": communication,
    "Aptitude_Test_Score": aptitude, "Soft_Skills_Rating": communication,
    "Certifications": certifications, "Backlogs": backlogs,
}
input_df        = pd.DataFrame([student_data])
prediction      = model.predict(input_df)[0]
probability     = model.predict_proba(input_df)[0][1]
readiness_score = calculate_readiness_score(student_data)
readiness_label = get_readiness_label(readiness_score)
pred_text       = "Placed" if prediction == 1 else "Not Placed"
prob_pct        = f"{probability * 100:.0f}%"

# ── STAT CARDS ──
st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card green">
        <div class="stat-label">Readiness Score</div>
        <div class="stat-value">{readiness_score}<span style="font-size:1rem;font-weight:400;color:var(--muted)">/100</span></div>
        <div class="stat-note">Overall placement preparedness</div>
    </div>
    <div class="stat-card blue">
        <div class="stat-label">ML Prediction</div>
        <div class="stat-value" style="font-size:1.5rem">{pred_text}</div>
        <div class="stat-note">Model confidence: {prob_pct}</div>
    </div>
    <div class="stat-card amber">
        <div class="stat-label">Industry Status</div>
        <div class="stat-value" style="font-size:1.3rem">{readiness_label}</div>
        <div class="stat-note">vs. market expectations</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── RADAR CHART — replaces barpolar for better multi-axis comparison ──
st.markdown('<div class="section-heading">Skills Radar — You vs Benchmark</div>', unsafe_allow_html=True)

benchmarks = {
    "Data Scientist":    {"cgpa": 7.5, "skills": 10, "projects": 3, "internships": 2, "certs": 2},
    "Software Engineer": {"cgpa": 7.0, "skills":  8, "projects": 3, "internships": 1, "certs": 1},
    "ML Engineer":       {"cgpa": 8.0, "skills": 12, "projects": 3, "internships": 2, "certs": 3},
    "Frontend Developer":{"cgpa": 6.5, "skills":  7, "projects": 3, "internships": 1, "certs": 1},
}
bm = benchmarks.get(target_role, benchmarks["Software Engineer"])

# Normalise everything to 0–10 scale for radar
categories = ["CGPA", "Coding Skills", "Aptitude", "Projects", "Internships", "Certifications", "Communication"]
you_raw    = [cgpa, min(len(skills), 10), aptitude / 10, min(projects * 2, 10),
              min(internships * 3, 10), min(certifications * 2.5, 10), communication]
bench_raw  = [bm["cgpa"], bm["skills"], 7.0, bm["projects"] * 2,
              bm["internships"] * 3, bm["certs"] * 2.5, 7.5]

# Close the polygon
cats_closed  = categories + [categories[0]]
you_closed   = you_raw + [you_raw[0]]
bench_closed = bench_raw + [bench_raw[0]]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=bench_closed, theta=cats_closed,
    fill='toself',
    name=f'{target_role} Benchmark',
    fillcolor='rgba(99,130,255,0.10)',
    line=dict(color='rgba(99,130,255,0.45)', width=1.5, dash='dot'),
    hovertemplate="<b>%{theta}</b><br>Benchmark: %{r:.1f}/10<extra></extra>"
))

fig.add_trace(go.Scatterpolar(
    r=you_closed, theta=cats_closed,
    fill='toself',
    name='Your Profile',
    fillcolor='rgba(52,211,153,0.15)',
    line=dict(color='#34d399', width=2.5),
    marker=dict(color='#34d399', size=7),
    hovertemplate="<b>%{theta}</b><br>Your score: %{r:.1f}/10<extra></extra>"
))

fig.update_layout(
    polar=dict(
        bgcolor='#080f1e',
        radialaxis=dict(
            visible=True, range=[0, 10],
            tickvals=[2, 4, 6, 8, 10],
            tickfont=dict(color='rgba(99,130,255,0.5)', size=10, family='DM Sans'),
            gridcolor='rgba(99,130,255,0.12)',
            linecolor='rgba(99,130,255,0.15)',
            ticksuffix='',
        ),
        angularaxis=dict(
            tickfont=dict(size=13, color='#a5b4fc', family='DM Sans'),
            gridcolor='rgba(99,130,255,0.10)',
            linecolor='rgba(99,130,255,0.15)',
        )
    ),
    legend=dict(
        orientation='h', yanchor='bottom', y=-0.18, xanchor='center', x=0.5,
        font=dict(size=12, color='#a5b4fc', family='DM Sans'),
        bgcolor='rgba(0,0,0,0)'
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    template='plotly_dark',
    height=480,
    margin=dict(t=30, b=60, l=60, r=60),
    showlegend=True,
)
st.plotly_chart(fig, use_container_width=True)

# ── SCORE BREAKDOWN ──
st.markdown('<div class="section-heading">Score Breakdown</div>', unsafe_allow_html=True)

def bar_color(val, max_val):
    pct = val / max_val
    if pct >= 0.8: return "#34d399"
    if pct >= 0.5: return "#fbbf24"
    return "#f87171"

breakdown = [
    ("CGPA",          cgpa,          10,   f"{cgpa:.1f}/10"),
    ("Communication", communication, 10,   f"{communication}/10"),
    ("Aptitude",      aptitude,      100,  f"{aptitude}/100"),
    ("Projects",      projects,      5,    f"{projects} projects"),
    ("Internships",   internships,   3,    f"{internships} internships"),
    ("Certifications",certifications,4,    f"{certifications} certs"),
]

bd_html = '<div class="breakdown-grid">'
for label, val, mx, disp in breakdown:
    pct = min((val / mx) * 100, 100)
    col = bar_color(val, mx)
    bd_html += f"""
<div class="breakdown-item">
    <div class="bi-header"><div class="bi-label">{label}</div><div class="bi-val">{disp}</div></div>
    <div class="bi-bar-bg"><div class="bi-bar-fill" style="width:{pct:.0f}%;background:{col};"></div></div>
</div>"""
bd_html += '</div>'
st.markdown(bd_html, unsafe_allow_html=True)

# ── BENCHMARK ──
def bench_color(you, need):
    return "#34d399" if you >= need else ("#fbbf24" if you >= need * 0.75 else "#f87171")

st.markdown(f'<div class="section-heading">Your Profile vs {target_role} Benchmark</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="bench-grid">
    <div class="bench-card">
        <div class="bench-title">CGPA</div>
        <div class="bench-row"><div class="bench-label">You</div><div class="bench-you" style="color:{bench_color(cgpa,bm['cgpa'])}">{cgpa:.1f}</div></div>
        <div class="bench-row"><div class="bench-label">Required</div><div class="bench-you" style="color:var(--muted)">{bm['cgpa']}</div></div>
        <div class="bench-bar-wrap"><div class="bench-bar" style="width:{min((cgpa/bm['cgpa'])*100,100):.0f}%;background:{bench_color(cgpa,bm['cgpa'])};"></div></div>
    </div>
    <div class="bench-card">
        <div class="bench-title">Technical Skills</div>
        <div class="bench-row"><div class="bench-label">You</div><div class="bench-you" style="color:{bench_color(len(skills),bm['skills'])}">{len(skills)}</div></div>
        <div class="bench-row"><div class="bench-label">Required</div><div class="bench-you" style="color:var(--muted)">{bm['skills']}</div></div>
        <div class="bench-bar-wrap"><div class="bench-bar" style="width:{min((len(skills)/bm['skills'])*100,100):.0f}%;background:{bench_color(len(skills),bm['skills'])};"></div></div>
    </div>
    <div class="bench-card">
        <div class="bench-title">Projects</div>
        <div class="bench-row"><div class="bench-label">You</div><div class="bench-you" style="color:{bench_color(projects,bm['projects'])}">{projects}</div></div>
        <div class="bench-row"><div class="bench-label">Required</div><div class="bench-you" style="color:var(--muted)">{bm['projects']}</div></div>
        <div class="bench-bar-wrap"><div class="bench-bar" style="width:{min((projects/bm['projects'])*100,100):.0f}%;background:{bench_color(projects,bm['projects'])};"></div></div>
    </div>
    <div class="bench-card">
        <div class="bench-title">Internships</div>
        <div class="bench-row"><div class="bench-label">You</div><div class="bench-you" style="color:{bench_color(internships,bm['internships'])}">{internships}</div></div>
        <div class="bench-row"><div class="bench-label">Required</div><div class="bench-you" style="color:var(--muted)">{bm['internships']}</div></div>
        <div class="bench-bar-wrap"><div class="bench-bar" style="width:{min((max(internships,0.01)/bm['internships'])*100,100):.0f}%;background:{bench_color(internships,bm['internships'])};"></div></div>
    </div>
    <div class="bench-card">
        <div class="bench-title">Certifications</div>
        <div class="bench-row"><div class="bench-label">You</div><div class="bench-you" style="color:{bench_color(certifications,bm['certs'])}">{certifications}</div></div>
        <div class="bench-row"><div class="bench-label">Required</div><div class="bench-you" style="color:var(--muted)">{bm['certs']}</div></div>
        <div class="bench-bar-wrap"><div class="bench-bar" style="width:{min((max(certifications,0.01)/bm['certs'])*100,100):.0f}%;background:{bench_color(certifications,bm['certs'])};"></div></div>
    </div>
    <div class="bench-card">
        <div class="bench-title">Aptitude Score</div>
        <div class="bench-row"><div class="bench-label">You</div><div class="bench-you" style="color:{bench_color(aptitude,70)}">{aptitude}</div></div>
        <div class="bench-row"><div class="bench-label">Target</div><div class="bench-you" style="color:var(--muted)">70</div></div>
        <div class="bench-bar-wrap"><div class="bench-bar" style="width:{min((aptitude/70)*100,100):.0f}%;background:{bench_color(aptitude,70)};"></div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TECH STACK ──
st.markdown('<div class="section-heading">Tech Stack Detected</div>', unsafe_allow_html=True)
if skills:
    st.markdown('<div class="chip-grid">' + ''.join(f'<div class="chip">{s}</div>' for s in skills) + '</div>', unsafe_allow_html=True)
else:
    st.warning("No skills detected from resume.")

# ── FEATURES ──
st.markdown('<div class="section-heading">Resume Extracted Features</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="metric-row">
    <div class="metric-box"><div class="metric-num">{internships}</div><div class="metric-lbl">Internships</div></div>
    <div class="metric-box"><div class="metric-num">{projects}</div><div class="metric-lbl">Projects</div></div>
    <div class="metric-box"><div class="metric-num">{certifications}</div><div class="metric-lbl">Certifications</div></div>
</div>
""", unsafe_allow_html=True)

# ── NEXT STEPS ──
st.markdown('<div class="section-heading">Recommended Next Steps</div>', unsafe_allow_html=True)
steps = []
if cgpa < bm["cgpa"]: steps.append(("Improve CGPA", f"Your CGPA {cgpa:.1f} is below the {target_role} benchmark of {bm['cgpa']}. Focus on upcoming exams."))
if len(skills) < bm["skills"]: steps.append(("Add More Skills", f"You have {len(skills)} skills but {target_role} roles expect {bm['skills']}+. Check the skill gap report."))
if internships < bm["internships"]: steps.append(("Get Internship Experience", "Look for virtual or part-time opportunities on LinkedIn, Internshala, or company portals."))
if projects < bm["projects"]: steps.append(("Build More Projects", f"You have {projects} project(s), target is {bm['projects']}+. Host on GitHub."))
if certifications < bm["certs"]: steps.append(("Earn Certifications", f"Add {bm['certs'] - certifications} more cert(s) from Coursera, NPTEL, or Udemy."))
if aptitude < 70: steps.append(("Improve Aptitude", f"Score of {aptitude}/100 is below 70 target. Practice on IndiaBix or PrepInsta."))
if not steps:
    steps = [
        ("Maintain Consistency", "Your profile meets benchmarks well. Keep skills updated."),
        ("Prepare for Interviews", "Practice on LeetCode, GeeksForGeeks, and Glassdoor."),
        ("Network Actively", "Connect on LinkedIn, attend virtual job fairs."),
    ]
for tag, text in steps[:3]:
    st.markdown(f'<div class="next-card"><div class="next-tag">{tag}</div><div class="next-text">{text}</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
if st.button("View Full Skill Gap Analysis →"):
    st.switch_page("pages/skill_gap.py")