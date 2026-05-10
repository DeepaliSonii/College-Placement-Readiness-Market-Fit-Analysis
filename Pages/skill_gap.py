import streamlit as st
import plotly.graph_objects as go

from skill_gap_llm import cached_skill_gap

# ── PAGE CONFIG (must be first st call) ─────────────────────────────────────
st.set_page_config(
    page_title="Skill Gap Analysis · PlaceIQ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>[data-testid="stSidebarNav"]{display:none;}</style>
""", unsafe_allow_html=True)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root{
    --bg:#040810; --surface:#080f1e; --surface-2:#0c1526;
    --border:rgba(99,130,255,0.12); --text:#e8eeff;
    --muted:#6b7a9e; --accent:#6382ff;
    --green:#34d399; --amber:#fbbf24; --red:#f87171;
}

#MainMenu,footer,header{visibility:hidden;}
.stApp{background:var(--bg);font-family:'DM Sans',sans-serif;color:var(--text);}
.block-container{max-width:1200px;padding:2rem 2rem 4rem;}

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
.sb-divider{height:1px;background:rgba(99,130,255,0.08);margin:0.8rem 1rem;}
.sb-footer{padding:0.8rem 1.2rem;font-size:0.7rem;color:rgba(107,122,158,0.45);line-height:1.55;}

.page-header{margin-bottom:2rem;}
.header-eyebrow{font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--accent);margin-bottom:0.4rem;}
.page-title{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#f0f4ff;margin:0 0 0.3rem;}
.page-subtitle{color:var(--muted);font-size:0.85rem;font-weight:300;}

.stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:1.5rem 0 2rem;}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.2rem 1.3rem;position:relative;overflow:hidden;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.stat-card.green::before{background:linear-gradient(90deg,#34d399,transparent);}
.stat-card.amber::before{background:linear-gradient(90deg,#fbbf24,transparent);}
.stat-card.red::before{background:linear-gradient(90deg,#f87171,transparent);}
.stat-label{font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);margin-bottom:0.5rem;}
.stat-value{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;line-height:1;}
.stat-card.green .stat-value{color:#34d399;}
.stat-card.amber .stat-value{color:#fbbf24;}
.stat-card.red   .stat-value{color:#f87171;}
.stat-note{font-size:0.73rem;color:var(--muted);margin-top:3px;}

.section-heading{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f4ff;margin:1.8rem 0 1rem;display:flex;align-items:center;gap:10px;}
.section-heading::after{content:'';flex:1;height:1px;background:var(--border);}

.chip-grid{display:flex;flex-wrap:wrap;gap:10px;margin:0.5rem 0 1.5rem;}
.chip-green{background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.25);color:#34d399;border-radius:8px;padding:8px 18px;font-size:0.85rem;font-weight:500;}
.chip-red{background:rgba(248,113,113,0.07);border:1px solid rgba(248,113,113,0.25);color:#f87171;border-radius:8px;padding:8px 18px;font-size:0.85rem;font-weight:500;display:flex;align-items:center;gap:6px;}
.priority-badge{background:rgba(248,113,113,0.15);border-radius:4px;padding:2px 6px;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;color:#fca5a5;}

.gap-card{background:var(--surface);border:1px solid var(--border);border-left:3px solid #f87171;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:12px;}
.gap-title{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f4ff;margin-bottom:10px;display:flex;align-items:center;gap:10px;}
.gap-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;}
.gap-field{font-size:0.8rem;}
.gap-field-label{color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;font-size:0.7rem;margin-bottom:2px;}
.gap-field-value{color:var(--text);}

.advice-card{background:linear-gradient(135deg,rgba(99,130,255,0.06),rgba(167,139,250,0.04));border:1px solid rgba(99,130,255,0.2);border-radius:14px;padding:1.4rem 1.5rem;margin:0.5rem 0 1.5rem;color:#c7d2fe;font-size:0.9rem;line-height:1.7;}
.roadmap-card{background:linear-gradient(135deg,rgba(52,211,153,0.05),rgba(6,182,212,0.03));border:1px solid rgba(52,211,153,0.18);border-radius:14px;padding:1.4rem 1.5rem;margin:0.5rem 0 1.5rem;color:#a7f3d0;font-size:0.9rem;line-height:1.7;}

.timeline{position:relative;padding-left:2rem;margin:0.5rem 0 1.5rem;}
.timeline::before{content:'';position:absolute;left:7px;top:4px;bottom:4px;width:1px;background:var(--border);}
.tl-item{position:relative;margin-bottom:1.2rem;}
.tl-dot{position:absolute;left:-1.6rem;top:4px;width:10px;height:10px;border-radius:50%;border:2px solid var(--accent);background:var(--bg);}
.tl-week{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--accent);margin-bottom:3px;}
.tl-title{font-family:'Syne',sans-serif;font-size:0.88rem;font-weight:700;color:#e8eeff;margin-bottom:3px;}
.tl-desc{font-size:0.78rem;color:var(--muted);line-height:1.5;}

.res-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:0.5rem 0 1.5rem;}
.res-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.1rem;}
.res-platform{font-size:0.67rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--accent);margin-bottom:0.3rem;}
.res-name{font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;color:#e8eeff;margin-bottom:0.25rem;}
.res-desc{font-size:0.75rem;color:var(--muted);line-height:1.5;}
.res-tag{display:inline-block;background:rgba(99,130,255,0.08);border:1px solid rgba(99,130,255,0.15);border-radius:5px;padding:2px 8px;font-size:0.68rem;color:#a5b4fc;margin-top:6px;}

.stButton>button{background:var(--surface);color:var(--accent) !important;border:1px solid rgba(99,130,255,0.3) !important;border-radius:12px;padding:0.75rem 1.5rem;font-family:'Syne',sans-serif;font-size:0.88rem;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;transition:all 0.2s;}
.stButton>button:hover{background:rgba(99,130,255,0.08) !important;transform:translateY(-1px);}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""

<div class="sb-section-label">Main Menu</div>
""", unsafe_allow_html=True)
    if st.button("🏠  Home",                key="sb_home"): st.switch_page("app.py")
    if st.button("📊  Readiness Dashboard", key="sb_rd"):   st.switch_page("pages/readiness.py")
    st.markdown('<div class="sb-nav active"><span class="sb-icon">🔍</span> Skill Gap Analysis</div>', unsafe_allow_html=True)
    if st.button("✏️   Quick Test",           key="sb_qz"):   st.switch_page("pages/quiz.py")
    st.markdown('<div class="sb-divider"></div><div class="sb-footer">PlaceIQ v2.0<br>AI · ML · Career Intelligence</div>', unsafe_allow_html=True)

# ── SESSION GUARD ─────────────────────────────────────────────────────────────
if "skills" not in st.session_state or "target_role" not in st.session_state:
    st.error("Session expired. Please return to Home and re-upload your resume.")
    if st.button("← Go Home"):
        st.switch_page("app.py")
    st.stop()

skills      = st.session_state.skills
target_role = st.session_state.target_role

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div class="header-eyebrow">Skill · Gap Analysis</div>
    <div class="page-title">Market Fit Analysis</div>
    <div class="page-subtitle">Your skills vs. industry requirements for <strong style="color:#e8eeff">{target_role}</strong></div>
</div>
""", unsafe_allow_html=True)

# ── FETCH (cached per role+skills — no repeat API calls on re-render) ────────
skills_key = ",".join(sorted(skills))
with st.spinner("Analysing your skill profile against industry benchmarks…"):
    data = cached_skill_gap(target_role, skills_key)

if not data:
    st.error("Analysis failed. Please return and try again.")
    if st.button("← Go Back"):
        st.switch_page("pages/readiness.py")
    st.stop()

market_fit_score    = int(data.get("market_fit_score", 0))
matched_skills      = data.get("matched_skills", [])
gap_skills          = data.get("gap_skills", [])
required_skills     = data.get("required_skills", [])
personalized_advice = data.get("personalized_advice", "")
learning_roadmap    = data.get("learning_roadmap", "")

# ── STAT CARDS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card green">
        <div class="stat-label">Market Fit Score</div>
        <div class="stat-value">{market_fit_score}%</div>
        <div class="stat-note">Skills matched vs. role requirements</div>
    </div>
    <div class="stat-card amber">
        <div class="stat-label">Strong Skills</div>
        <div class="stat-value">{len(matched_skills)}</div>
        <div class="stat-note">Already job-ready</div>
    </div>
    <div class="stat-card red">
        <div class="stat-label">Skill Gaps</div>
        <div class="stat-value">{len(gap_skills)}</div>
        <div class="stat-note">Areas to develop</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FIT PROGRESS BAR ─────────────────────────────────────────────────────────
fit_color = "#34d399" if market_fit_score >= 70 else ("#fbbf24" if market_fit_score >= 40 else "#f87171")
st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.4rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;">Overall Market Fit</span>
        <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:{fit_color};">{market_fit_score}%</span>
    </div>
    <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">
        <div style="height:8px;width:{market_fit_score}%;background:linear-gradient(90deg,{fit_color},{fit_color}88);border-radius:99px;"></div>
    </div>
    <div style="font-size:0.75rem;color:var(--muted);margin-top:6px;">{len(matched_skills)} of {len(required_skills)} required skills matched</div>
</div>
""", unsafe_allow_html=True)

# ── LOLLIPOP CHART — Your Skills vs. Market Requirements ─────────────────────
st.markdown('<div class="section-heading">Your Skills vs. Market Requirements</div>', unsafe_allow_html=True)

matched_lower = [s.lower() for s in matched_skills]
your_scores   = [10 if s.lower() in matched_lower else 0 for s in required_skills]
market_scores = [10] * len(required_skills)

def shorten(name, n=22):
    return name if len(name) <= n else name[:n-1] + "…"

labels     = [shorten(s) for s in required_skills]
dot_colors = ["#34d399" if v == 10 else "#f87171" for v in your_scores]
gap_values = [m - y for m, y in zip(market_scores, your_scores)]  # always 0 or 10

fig = go.Figure()

# ── horizontal base line at y=0 for each skill (the "stem" base) ──────────────
for i, (label, your_val, dot_col) in enumerate(zip(labels, your_scores, dot_colors)):
    # Stem from 0 → your score (coloured)
    fig.add_shape(
        type="line",
        x0=i, x1=i, y0=0, y1=your_val,
        line=dict(color=dot_col, width=3),
    )
    # Stem from your score → market (grey gap, only when skill is missing)
    if your_val < 10:
        fig.add_shape(
            type="line",
            x0=i, x1=i, y0=your_val, y1=10,
            line=dict(color="rgba(99,130,255,0.25)", width=2, dash="dot"),
        )

# ── Market requirement dots (top, semi-transparent) ───────────────────────────
fig.add_trace(go.Scatter(
    x=list(range(len(labels))),
    y=market_scores,
    mode="markers",
    name="Market Requirement",
    marker=dict(
        symbol="circle",
        size=14,
        color="rgba(99,130,255,0.35)",
        line=dict(color="rgba(99,130,255,0.70)", width=2),
    ),
    hovertemplate="<b>%{customdata}</b><br>Required: 10 / 10<extra></extra>",
    customdata=labels,
))

# ── Your proficiency dots ──────────────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=list(range(len(labels))),
    y=your_scores,
    mode="markers+text",
    name="Your Proficiency",
    marker=dict(
        symbol="circle",
        size=18,
        color=dot_colors,
        line=dict(color="rgba(255,255,255,0.20)", width=2),
    ),
    text=["✓" if v == 10 else "✗" for v in your_scores],
    textposition="middle center",
    textfont=dict(size=10, color="#0c1526"),
    hovertemplate="<b>%{customdata}</b><br>Your score: %{y} / 10<extra></extra>",
    customdata=labels,
))

fig.update_layout(
    template="plotly_dark",
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=30, b=60, l=10, r=10),
    font=dict(family="DM Sans", color="#a5b4fc"),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=12, color="#a5b4fc"), bgcolor="rgba(0,0,0,0)"
    ),
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=labels,
        tickfont=dict(size=11, color="#a5b4fc"),
        tickangle=-30,
        gridcolor="rgba(99,130,255,0.06)",
        linecolor="rgba(99,130,255,0.12)",
        showgrid=False,
    ),
    yaxis=dict(
        range=[-0.5, 12],
        tickvals=[0, 5, 10],
        ticktext=["0", "5", "10"],
        tickfont=dict(size=11, color="#6b7a9e"),
        gridcolor="rgba(99,130,255,0.08)",
        linecolor="rgba(99,130,255,0.12)",
        title=dict(text="Proficiency (0 – 10)", font=dict(size=11, color="#6b7a9e")),
        zeroline=True,
        zerolinecolor="rgba(99,130,255,0.18)",
        zerolinewidth=1,
    ),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="display:flex;gap:24px;justify-content:center;margin-bottom:1rem;margin-top:-0.5rem;">
    <span style="font-size:0.78rem;color:#34d399;">● Skill matched</span>
    <span style="font-size:0.78rem;color:#f87171;">● Skill gap</span>
    <span style="font-size:0.78rem;color:rgba(99,130,255,0.7);">● Market requirement</span>
</div>
""", unsafe_allow_html=True)

# ── SKILL-BY-SKILL BREAKDOWN ─────────────────────────────────────────────────
st.markdown('<div class="section-heading">Skill-by-Skill Breakdown</div>', unsafe_allow_html=True)
rows = ""
for skill, score in zip(required_skills, your_scores):
    col    = "#34d399" if score == 10 else "#f87171"
    status = "✓ Matched" if score == 10 else "✗ Gap"
    rows += f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:0.8rem 1.1rem;margin-bottom:8px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-size:0.85rem;font-weight:500;color:#e8eeff;">{skill}</span>
        <span style="font-size:0.75rem;font-weight:600;color:{col};">{status}</span>
    </div>
    <div style="height:5px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">
        <div style="height:5px;width:{score * 10}%;background:{col};border-radius:99px;"></div>
    </div>
</div>"""
st.markdown(rows, unsafe_allow_html=True)

# ── MATCHED / GAP CHIPS ───────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Matched Skills</div>', unsafe_allow_html=True)
if matched_skills:
    st.markdown('<div class="chip-grid">' + ''.join(f'<div class="chip-green">✓ {s}</div>' for s in matched_skills) + '</div>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:var(--muted);font-size:0.9rem;">No strong skill matches found — more room to grow!</p>', unsafe_allow_html=True)

st.markdown('<div class="section-heading">Skills to Develop</div>', unsafe_allow_html=True)
if gap_skills:
    st.markdown('<div class="chip-grid">' + ''.join(
        f'<div class="chip-red">✗ {g["skill"]} <span class="priority-badge">{g.get("priority","")}</span></div>'
        for g in gap_skills
    ) + '</div>', unsafe_allow_html=True)

# ── DETAILED GAP BREAKDOWN ────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Detailed Gap Breakdown</div>', unsafe_allow_html=True)
for gap in gap_skills:
    st.markdown(f"""
<div class="gap-card">
    <div class="gap-title">
        {gap['skill']}
        <span style="font-size:0.72rem;font-weight:400;background:rgba(248,113,113,0.1);color:#fca5a5;padding:3px 10px;border-radius:5px;letter-spacing:0.06em;text-transform:uppercase;">{gap.get('priority','')}</span>
    </div>
    <div class="gap-row">
        <div class="gap-field"><div class="gap-field-label">Why Important</div><div class="gap-field-value">{gap['reason']}</div></div>
        <div class="gap-field"><div class="gap-field-label">Recommended Resource</div><div class="gap-field-value">{gap['course']}</div></div>
    </div>
</div>""", unsafe_allow_html=True)

# ── LEARNING TIMELINE ─────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Suggested Learning Timeline</div>', unsafe_allow_html=True)
high_prio   = [g for g in gap_skills if str(g.get("priority","")).upper() == "HIGH"][:2]
medium_prio = [g for g in gap_skills if str(g.get("priority","")).upper() == "MEDIUM"][:2]
low_prio    = [g for g in gap_skills if str(g.get("priority","")).upper() == "LOW"][:1]

tl_items = []
if high_prio:   tl_items.append(("Week 1–3",   "Foundation",      f"Start with: {', '.join(g['skill'] for g in high_prio)}. Core requirements for {target_role}."))
if medium_prio: tl_items.append(("Week 4–6",   "Build Depth",     f"Build on: {', '.join(g['skill'] for g in medium_prio)}. Complete a guided project using these."))
if low_prio:    tl_items.append(("Week 7–8",   "Polish & Apply",  f"Cover: {', '.join(g['skill'] for g in low_prio)} and update your portfolio."))
tl_items.append(("Week 9–10",  "Mock Interviews", f"Practice {target_role} interview questions on LeetCode and GeeksForGeeks."))
tl_items.append(("Week 11–12", "Apply Actively",  "Start applying via LinkedIn, Naukri, InternShala. Target dream and backup companies."))

st.markdown('<div class="timeline">', unsafe_allow_html=True)
for week, title, desc in tl_items:
    st.markdown(f'<div class="tl-item"><div class="tl-dot"></div><div class="tl-week">{week}</div><div class="tl-title">{title}</div><div class="tl-desc">{desc}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── FREE RESOURCES ────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Free Learning Resources</div>', unsafe_allow_html=True)
role_resources = {
    "Data Scientist": [
        ("Coursera", "IBM Data Science Professional Certificate", "Python, ML, SQL and data visualisation end-to-end.", "Free Audit"),
        ("Kaggle", "Kaggle Learn — Micro-Courses", "Pandas, ML, feature engineering and more.", "Free"),
        ("YouTube", "StatQuest with Josh Starmer", "Visual explanations of statistics and ML algorithms.", "Free"),
    ],
    "Software Engineer": [
        ("LeetCode", "Top 150 Interview Questions", "DSA problems: arrays, trees, graphs, DP.", "Free Tier"),
        ("GeeksForGeeks", "DSA Self-Paced Course", "Full data structures and algorithms course.", "Free"),
        ("freeCodeCamp", "Full Stack Developer Path", "HTML, CSS, JS, React, Node, and databases.", "Free"),
    ],
    "ML Engineer": [
        ("Fast.ai", "Practical Deep Learning for Coders", "Hands-on PyTorch-based deep learning.", "Free"),
        ("Coursera", "MLOps Specialization — Andrew Ng", "ML deployment, monitoring, production pipelines.", "Free Audit"),
        ("Hugging Face", "NLP Course", "Official transformer and LLM training course.", "Free"),
    ],
    "Frontend Developer": [
        ("freeCodeCamp", "Responsive Web Design Cert", "HTML, CSS and modern layout — project based.", "Free"),
        ("The Odin Project", "Full Stack JavaScript Path", "Open-source curriculum from basics to React.", "Free"),
        ("Scrimba", "Learn React for Free", "Interactive in-browser React course.", "Free"),
    ],
}
resources = role_resources.get(target_role, role_resources["Software Engineer"])
st.markdown('<div class="res-grid">', unsafe_allow_html=True)
for platform, name, desc, tag in resources:
    st.markdown(f'<div class="res-card"><div class="res-platform">{platform}</div><div class="res-name">{name}</div><div class="res-desc">{desc}</div><div class="res-tag">{tag}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── ADVICE & ROADMAP ──────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Personalised Career Advice</div>', unsafe_allow_html=True)
st.markdown(f'<div class="advice-card">{personalized_advice}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-heading">AI-Generated Learning Roadmap</div>', unsafe_allow_html=True)
st.markdown(f'<div class="roadmap-card">{learning_roadmap}</div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
if st.button("← Return to Readiness Dashboard"):
    st.switch_page("pages/readiness.py")