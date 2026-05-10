import random
import time
import streamlit as st

st.set_page_config(
    page_title="Quick Test · PlaceIQ",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>[data-testid="stSidebarNav"]{display:none;}</style>
""", unsafe_allow_html=True)

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#040810;--surface:#080f1e;--surface-2:#0c1526;--border:rgba(99,130,255,0.12);--text:#e8eeff;--muted:#6b7a9e;--accent:#6382ff;--accent-2:#a78bfa;--green:#34d399;--amber:#fbbf24;--red:#f87171;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:var(--bg);font-family:'DM Sans',sans-serif;color:var(--text);}
.block-container{max-width:900px;padding:2rem 2rem 4rem;}
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
.page-header{margin-bottom:1.5rem;}
.header-eyebrow{font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--accent);margin-bottom:0.4rem;}
.page-title{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#f0f4ff;margin:0 0 0.3rem;}
.page-subtitle{color:var(--muted);font-size:0.85rem;font-weight:300;}
.subject-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:1.5rem 0;}
.subject-card{background:var(--surface);border:2px solid var(--border);border-radius:18px;padding:1.8rem;cursor:pointer;transition:all 0.2s;text-align:center;}
.subject-card:hover{border-color:rgba(99,130,255,0.45);background:rgba(99,130,255,0.05);transform:translateY(-3px);}
.sc-icon{font-size:2.8rem;margin-bottom:0.7rem;}
.sc-title{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;color:#f0f4ff;margin-bottom:0.4rem;}
.sc-desc{font-size:0.82rem;color:var(--muted);line-height:1.5;margin-bottom:0.8rem;}
.sc-marks{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;}
.mark-pill{background:rgba(99,130,255,0.08);border:1px solid rgba(99,130,255,0.2);border-radius:6px;padding:3px 10px;font-size:0.73rem;color:#a5b4fc;}
.quiz-progress{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.5rem;}
.qp-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
.qp-label{font-size:0.8rem;color:var(--muted);}
.qp-count{font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#e8eeff;}
.qp-bar-bg{height:6px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;}
.qp-bar-fill{height:6px;background:linear-gradient(90deg,#6382ff,#a78bfa);border-radius:99px;transition:width 0.4s;}
.qp-meta{display:flex;gap:16px;margin-top:8px;flex-wrap:wrap;}
.qp-meta-item{font-size:0.75rem;color:var(--muted);}
.qp-meta-item span{color:#e8eeff;font-weight:600;}
.q-card{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:1.8rem;margin-bottom:1.2rem;}
.q-difficulty{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;margin-bottom:0.7rem;}
.q-difficulty.easy{color:#34d399;}
.q-difficulty.medium{color:#fbbf24;}
.q-difficulty.hard{color:#f87171;}
.q-marks{background:rgba(99,130,255,0.08);border:1px solid rgba(99,130,255,0.2);border-radius:6px;padding:2px 10px;font-size:0.72rem;color:#a5b4fc;display:inline-block;margin-left:10px;}
.q-text{font-size:1rem;color:#f0f4ff;line-height:1.65;margin-bottom:1.2rem;font-weight:500;}
.q-code{background:#0a1020;border:1px solid rgba(99,130,255,0.15);border-radius:10px;padding:1rem;font-family:'Courier New',monospace;font-size:0.85rem;color:#a5b4fc;line-height:1.6;margin-bottom:1rem;white-space:pre-wrap;}
div[data-testid="stRadio"] label{background:var(--surface-2) !important;border:1px solid rgba(99,130,255,0.15) !important;border-radius:12px !important;padding:0.75rem 1rem !important;margin-bottom:8px !important;display:block;cursor:pointer;transition:all 0.18s;color:var(--text) !important;font-family:'DM Sans',sans-serif !important;}
div[data-testid="stRadio"] label:hover{border-color:rgba(99,130,255,0.4) !important;background:rgba(99,130,255,0.06) !important;}
.answer-correct{background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.3);border-radius:12px;padding:1rem 1.2rem;margin-top:0.8rem;}
.answer-wrong{background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.3);border-radius:12px;padding:1rem 1.2rem;margin-top:0.8rem;}
.answer-label{font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;margin-bottom:0.4rem;}
.answer-label.correct{color:#34d399;}
.answer-label.wrong{color:#f87171;}
.answer-explain{font-size:0.85rem;color:#a5b4fc;line-height:1.6;margin-top:0.5rem;}
.result-card{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:2.4rem;text-align:center;margin-bottom:1.5rem;}
.result-score{font-family:'Syne',sans-serif;font-size:4rem;font-weight:800;line-height:1;}
.result-max{font-size:1.5rem;color:var(--muted);}
.result-label{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;margin:0.5rem 0 1rem;}
.result-bar-bg{height:10px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;margin:0.8rem 0;}
.result-bar{height:10px;border-radius:99px;}
.score-breakdown{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:1.5rem 0;}
.sb-item{background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:1rem;text-align:center;}
.sb-num{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;margin-bottom:3px;}
.sb-lbl{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.07em;}
.stButton>button{background:linear-gradient(135deg,#4b6eff,#7c3aed);color:white !important;border:none !important;border-radius:12px;padding:0.8rem 1.5rem;font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;transition:all 0.2s;width:100%;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(99,130,255,0.35);}
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""

<div class="sb-section-label">Main Menu</div>
""", unsafe_allow_html=True)
    if st.button("🏠  Home",                key="sb_home"): st.switch_page("app.py")
    if st.button("📊  Readiness Dashboard", key="sb_rd"):   st.switch_page("pages/readiness.py")
    if st.button("🔍  Skill Gap Analysis",  key="sb_sg"):   st.switch_page("pages/skill_gap.py")
    st.markdown('<div class="sb-nav active"><span class="sb-icon">✏️</span> Quick Test</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div><div class="sb-footer">PlaceIQ v2.0<br>AI · ML · Career Intelligence</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# QUESTION BANKS — 20 questions each, shuffled per attempt
# ══════════════════════════════════════════════════════════

ALL_APTITUDE = [
    # ── EASY (1 mark each) ─────────────────────────────────
    {"text":"A train travels 360 km in 4 hours. What is its speed in km/h?","options":["80","90","100","72"],"answer":1,"difficulty":"easy","marks":1,"explanation":"Speed = Distance ÷ Time = 360 ÷ 4 = 90 km/h."},
    {"text":"If 6 workers finish a task in 8 days, how many days will 4 workers take?","options":["10","12","9","14"],"answer":1,"difficulty":"easy","marks":1,"explanation":"Total work = 6×8 = 48 unit-days. Days for 4 workers = 48÷4 = 12."},
    {"text":"What is 15% of 240?","options":["30","36","32","28"],"answer":1,"difficulty":"easy","marks":1,"explanation":"15% of 240 = 0.15×240 = 36."},
    {"text":"A number is increased by 20% and then decreased by 20%. What is the net change?","options":["No change","4% decrease","4% increase","2% decrease"],"answer":1,"difficulty":"easy","marks":1,"explanation":"x→1.2x→0.96x. Net = −4%."},
    {"text":"The ratio of boys to girls in a class is 3:2. If there are 30 students, how many are boys?","options":["12","15","18","20"],"answer":2,"difficulty":"easy","marks":1,"explanation":"Boys = (3/5)×30 = 18."},
    {"text":"If today is Wednesday, what day will it be 100 days from now?","options":["Friday","Saturday","Sunday","Monday"],"answer":0,"difficulty":"easy","marks":1,"explanation":"100 mod 7 = 2. Wednesday + 2 = Friday."},
    {"text":"A car covers 150 km at 50 km/h and the next 150 km at 75 km/h. What is the average speed for the whole journey?","options":["60 km/h","62.5 km/h","65 km/h","67.5 km/h"],"answer":0,"difficulty":"easy","marks":1,"explanation":"Total time = 150/50 + 150/75 = 3+2 = 5 hrs. Total dist = 300 km. Avg = 300/5 = 60 km/h."},
    {"text":"Find the simple interest on ₹5000 at 8% per annum for 3 years.","options":["₹1000","₹1200","₹1500","₹1800"],"answer":1,"difficulty":"easy","marks":1,"explanation":"SI = (P×R×T)/100 = (5000×8×3)/100 = ₹1200."},
    {"text":"A fruit seller buys 100 oranges at ₹5 each and sells them at ₹7 each. What is the profit percentage?","options":["30%","35%","40%","45%"],"answer":2,"difficulty":"easy","marks":1,"explanation":"Cost=₹500, SP=₹700. Profit=₹200. Profit%=(200/500)×100=40%."},
    {"text":"If 3x − 7 = 11, what is the value of x?","options":["4","5","6","7"],"answer":2,"difficulty":"easy","marks":1,"explanation":"3x=18 → x=6."},

    # ── MEDIUM (2 marks each) ──────────────────────────────
    {"text":"A shopkeeper marks a product at ₹500 and gives 20% discount. Cost price is ₹300. What is the profit %?","options":["25%","33.3%","30%","28%"],"answer":1,"difficulty":"medium","marks":2,"explanation":"SP=500×0.8=₹400. Profit=100. Profit%=(100/300)×100≈33.3%."},
    {"text":"In a class of 60 students, 40% are girls. If 10 more girls join, what % of the new total are girls?","options":["42%","44%","46%","48.6%"],"answer":3,"difficulty":"medium","marks":2,"explanation":"Girls=24+10=34 out of 70. 34/70×100≈48.6%."},
    {"text":"The HCF and LCM of two numbers are 12 and 180. If one number is 36, find the other.","options":["48","54","60","72"],"answer":2,"difficulty":"medium","marks":2,"explanation":"Other = (HCF×LCM)/one = (12×180)/36 = 60."},
    {"text":"A and B can complete a job in 12 and 15 days respectively. How long do they take together?","options":["6 days","6.5 days","6 days 18 hrs","None"],"answer":0,"difficulty":"medium","marks":2,"explanation":"Combined rate = 1/12+1/15 = 9/60 = 3/20. Time = 20/3 ≈ 6.67 days ≈ 6 days 16 hrs. Closest is 6 days."},
    {"text":"A sum doubles itself in 10 years under simple interest. What is the annual interest rate?","options":["8%","10%","12%","15%"],"answer":1,"difficulty":"medium","marks":2,"explanation":"SI=P in 10 years → P=P×r×10/100 → r=10%."},
    {"text":"Two numbers are in ratio 5:7. Their LCM is 140. What is their sum?","options":["48","72","60","84"],"answer":1,"difficulty":"medium","marks":2,"explanation":"Numbers = 5k and 7k. LCM = 35k = 140 → k=4. Sum = 20+28 = 48. (Answer 0 is 48).","answer":0},
    {"text":"A cistern is filled by pipe A in 10 hrs and emptied by pipe B in 15 hrs. Both are opened together. When will the cistern be full (starting empty)?","options":["25 hrs","30 hrs","20 hrs","35 hrs"],"answer":1,"difficulty":"medium","marks":2,"explanation":"Net rate = 1/10−1/15 = 1/30. Time = 30 hrs."},
    {"text":"If the price of petrol increases by 25%, by what % should consumption decrease to keep expenditure unchanged?","options":["20%","25%","15%","10%"],"answer":0,"difficulty":"medium","marks":2,"explanation":"Reduction = 25/(100+25)×100 = 20%."},

    # ── HARD (3 marks each) ────────────────────────────────
    {"text":"Two pipes A and B fill a tank in 15 and 20 hours. Pipe C empties it in 25 hours. All open together — when is the tank full?","options":["300/23 hrs","15 hrs","100/7 hrs","20 hrs"],"answer":0,"difficulty":"hard","marks":3,"explanation":"Net rate=1/15+1/20−1/25=(20+15−12)/300=23/300. Time=300/23≈13.04 hrs."},
    {"text":"A merchant sells two articles at ₹990 each. On one he gains 10% and on the other loses 10%. What is the overall gain or loss?","options":["No change","1% gain","1% loss","2% loss"],"answer":2,"difficulty":"hard","marks":3,"explanation":"CP1=990/1.1=900, CP2=990/0.9=1100. Total CP=2000, SP=1980. Loss=₹20 → 1% loss."},
]

ALL_CODING = [
    # ── EASY (1 mark each) ─────────────────────────────────
    {"text":"What is the output of the following Python code?","code":"x = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))","options":["3","4","Error","None"],"answer":1,"difficulty":"easy","marks":1,"explanation":"y=x points to the same list. Appending to y also affects x. len(x)=4."},
    {"text":"What is the time complexity of binary search?","options":["O(n)","O(n²)","O(log n)","O(n log n)"],"answer":2,"difficulty":"easy","marks":1,"explanation":"Binary search halves the search space each step → O(log n)."},
    {"text":"Which keyword is used to handle exceptions in Python?","options":["catch","except","handle","error"],"answer":1,"difficulty":"easy","marks":1,"explanation":"Python uses 'except' in try-except blocks to handle exceptions."},
    {"text":"What does SQL SELECT DISTINCT do?","options":["Returns all rows","Removes duplicate rows","Sorts results","Filters NULL values"],"answer":1,"difficulty":"easy","marks":1,"explanation":"DISTINCT eliminates duplicate rows from the result set."},
    {"text":"What is the output of: print(type([]))?","options":["<class 'tuple'>","<class 'dict'>","<class 'list'>","<class 'set'>"],"answer":2,"difficulty":"easy","marks":1,"explanation":"[] creates an empty list. type([]) returns <class 'list'>."},
    {"text":"Which data structure uses LIFO (Last In First Out)?","options":["Queue","Stack","LinkedList","Tree"],"answer":1,"difficulty":"easy","marks":1,"explanation":"A Stack operates on LIFO — the last element pushed is the first to be popped."},
    {"text":"What does the 'git commit' command do?","options":["Pushes code to remote","Stages changes","Saves changes to local repo history","Merges branches"],"answer":2,"difficulty":"easy","marks":1,"explanation":"git commit saves staged changes into the local repository history with a message."},
    {"text":"In Python, what is the result of 5 // 2?","options":["2.5","2","3","Error"],"answer":1,"difficulty":"easy","marks":1,"explanation":"// is floor division. 5 // 2 = 2 (integer quotient, floor rounded)."},
    {"text":"Which HTTP method is used to send data to a server to create/update a resource?","options":["GET","DELETE","POST","HEAD"],"answer":2,"difficulty":"easy","marks":1,"explanation":"POST sends data in the request body to create or update a resource on the server."},
    {"text":"What is the index of the first element in an array in most programming languages?","options":["1","-1","0","Depends on language"],"answer":2,"difficulty":"easy","marks":1,"explanation":"Most languages (C, Java, Python, JS) use 0-based indexing. The first element is at index 0."},

    # ── MEDIUM (2 marks each) ──────────────────────────────
    {"text":"What does the following code print?","code":"def f(x, lst=[]):\n    lst.append(x)\n    return lst\n\nprint(f(1))\nprint(f(2))\nprint(f(3))","options":["[1]\n[2]\n[3]","[1]\n[1, 2]\n[1, 2, 3]","Error","[1,2,3] three times"],"answer":1,"difficulty":"medium","marks":2,"explanation":"Python default mutable arguments are created once and persist. The same list is shared across all calls."},
    {"text":"Which is MOST efficient for an LRU cache with O(1) get and put?","options":["Array + Binary Search","HashMap + Doubly Linked List","Binary Search Tree","Stack"],"answer":1,"difficulty":"medium","marks":2,"explanation":"HashMap gives O(1) lookup; Doubly Linked List gives O(1) insert/delete for recency tracking."},
    {"text":"What is the space complexity of merge sort?","options":["O(1)","O(log n)","O(n)","O(n log n)"],"answer":2,"difficulty":"medium","marks":2,"explanation":"Merge sort requires O(n) auxiliary space for the temporary arrays during merging."},
    {"text":"What is the output of: print(0.1 + 0.2 == 0.3) in Python?","options":["True","False","Error","None"],"answer":1,"difficulty":"medium","marks":2,"explanation":"Floating point representation causes 0.1+0.2 to be 0.30000000000000004, not exactly 0.3."},
    {"text":"Which SQL clause filters groups after GROUP BY?","options":["WHERE","FILTER","HAVING","GROUP FILTER"],"answer":2,"difficulty":"medium","marks":2,"explanation":"HAVING filters grouped results, while WHERE filters individual rows before grouping."},
    {"text":"In OOP, what does 'polymorphism' mean?","options":["Hiding data","One class inheriting another","Same interface, different implementations","Code reuse via composition"],"answer":2,"difficulty":"medium","marks":2,"explanation":"Polymorphism allows the same method name to behave differently depending on the object type."},
    {"text":"What does the following Python snippet return?\n\nprint(list(map(lambda x: x**2, [1,2,3,4])))","options":["[1,4,9,16]","[2,4,6,8]","(1,4,9,16)","Error"],"answer":0,"difficulty":"medium","marks":2,"explanation":"map applies the lambda (x²) to each element. list() converts the result to [1, 4, 9, 16]."},
    {"text":"What is a deadlock in operating systems?","options":["Infinite loop in a program","Two processes waiting for each other's resources indefinitely","Memory overflow","CPU starvation"],"answer":1,"difficulty":"medium","marks":2,"explanation":"A deadlock occurs when two or more processes each hold resources and wait for the other's, creating a circular wait."},

    # ── HARD (3 marks each) ────────────────────────────────
    {"text":"What is the output of the following generator code?","code":"def gen():\n    yield 1\n    yield 2\n    yield 3\n\ng = gen()\nprint(next(g))\nprint(next(g))\nlist(g)\nprint(next(g, 'done'))","options":["1\n2\n3","1\n2\ndone","1\n2\nError","1\n1\ndone"],"answer":1,"difficulty":"hard","marks":3,"explanation":"next(g) yields 1, then 2. list(g) exhausts remaining (yields 3). next(g,'done') returns default 'done' since generator is exhausted."},
    {"text":"Given an array of n integers, what is the best time complexity to find all pairs that sum to a target value k?","options":["O(n²)","O(n log n)","O(n)","O(n log k)"],"answer":2,"difficulty":"hard","marks":3,"explanation":"Use a HashSet: for each element x, check if (k−x) is in the set. Single pass → O(n) time, O(n) space."},
]

# ── SESSION STATE ──
for key, val in [
    ("quiz_state","select"),("quiz_subject",None),("quiz_questions",[]),
    ("quiz_current",0),("quiz_answers",{}),("quiz_score",0),
    ("quiz_answered",False),("quiz_selected",None),("quiz_start_time",None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

def pick_questions(pool):
    """Shuffle pool and pick a balanced set: 4 easy + 4 medium + 2 hard = 10 questions."""
    easy   = [q for q in pool if q["difficulty"] == "easy"]
    medium = [q for q in pool if q["difficulty"] == "medium"]
    hard   = [q for q in pool if q["difficulty"] == "hard"]
    random.shuffle(easy)
    random.shuffle(medium)
    random.shuffle(hard)
    selected = easy[:4] + medium[:4] + hard[:2]
    random.shuffle(selected)
    return selected

# ── HEADER ──
st.markdown("""
<div class="page-header">
    <div class="header-eyebrow">Placement · Assessment</div>
    <div class="page-title">Quick Test</div>
    <div class="page-subtitle">10 shuffled questions each attempt · Mixed difficulty · Placement-level.</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════
# STATE: SELECT SUBJECT
# ════════════════════════════════════
if st.session_state.quiz_state == "select":
    st.markdown("""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;">
    <div style="font-family:'Syne',sans-serif;font-size:0.8rem;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">Exam Structure</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
        <div style="text-align:center;"><div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#e8eeff;">10</div><div style="font-size:0.75rem;color:var(--muted);">Questions</div></div>
        <div style="text-align:center;"><div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#e8eeff;">20+</div><div style="font-size:0.75rem;color:var(--muted);">Question Bank</div></div>
        <div style="text-align:center;"><div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#e8eeff;">🔀</div><div style="font-size:0.75rem;color:var(--muted);">Shuffled</div></div>
        <div style="text-align:center;"><div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#e8eeff;">~15 min</div><div style="font-size:0.75rem;color:var(--muted);">Est. Time</div></div>
    </div>
    <div style="margin-top:1rem;font-size:0.8rem;color:var(--muted);line-height:1.6;">
        <b style="color:#a5b4fc;">Marking:</b> 4 Easy × 1 mark · 4 Medium × 2 marks · 2 Hard × 3 marks = <b style="color:#e8eeff;">18 total marks</b> · Questions shuffled every attempt.
    </div>
</div>
<div style="font-family:'Syne',sans-serif;font-size:0.8rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:1rem;">Choose Your Subject</div>
<div class="subject-grid">
    <div class="subject-card">
        <div class="sc-icon">🧮</div>
        <div class="sc-title">Quantitative Aptitude</div>
        <div class="sc-desc">Arithmetic, percentages, time &amp; work, speed &amp; distance, profit &amp; loss — essential for campus placements.</div>
        <div class="sc-marks">
            <div class="mark-pill">20 question bank</div>
            <div class="mark-pill">Shuffled per attempt</div>
        </div>
    </div>
    <div class="subject-card">
        <div class="sc-icon">💻</div>
        <div class="sc-title">Coding &amp; CS Fundamentals</div>
        <div class="sc-desc">Python/Java, data structures, algorithms, time complexity, OS, DBMS — core technical interview prep.</div>
        <div class="sc-marks">
            <div class="mark-pill">20 question bank</div>
            <div class="mark-pill">Shuffled per attempt</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧮  Start Aptitude Test", key="start_apt"):
            st.session_state.quiz_subject    = "Aptitude"
            st.session_state.quiz_questions  = pick_questions(ALL_APTITUDE)
            st.session_state.quiz_state      = "active"
            st.session_state.quiz_current    = 0
            st.session_state.quiz_answers    = {}
            st.session_state.quiz_score      = 0
            st.session_state.quiz_answered   = False
            st.session_state.quiz_selected   = None
            st.session_state.quiz_start_time = time.time()
            st.rerun()
    with col2:
        if st.button("💻  Start Coding Test", key="start_code"):
            st.session_state.quiz_subject    = "Coding"
            st.session_state.quiz_questions  = pick_questions(ALL_CODING)
            st.session_state.quiz_state      = "active"
            st.session_state.quiz_current    = 0
            st.session_state.quiz_answers    = {}
            st.session_state.quiz_score      = 0
            st.session_state.quiz_answered   = False
            st.session_state.quiz_selected   = None
            st.session_state.quiz_start_time = time.time()
            st.rerun()

# ════════════════════════════════════
# STATE: ACTIVE QUIZ
# ════════════════════════════════════
elif st.session_state.quiz_state == "active":
    questions = st.session_state.quiz_questions
    idx       = st.session_state.quiz_current
    total     = len(questions)
    q         = questions[idx]
    pct       = (idx / total) * 100
    total_marks = sum(x["marks"] for x in questions)

    elapsed = int(time.time() - st.session_state.quiz_start_time) if st.session_state.quiz_start_time else 0
    mins, secs = divmod(elapsed, 60)
    scored_so_far = sum(
        q2["marks"] for i2, q2 in enumerate(questions)
        if i2 in st.session_state.quiz_answers and st.session_state.quiz_answers[i2]["correct"]
    )

    st.markdown(f"""
<div class="quiz-progress">
    <div class="qp-header">
        <div class="qp-label">Question Progress</div>
        <div class="qp-count">{idx + 1} / {total}</div>
    </div>
    <div class="qp-bar-bg"><div class="qp-bar-fill" style="width:{pct:.0f}%;"></div></div>
    <div class="qp-meta">
        <div class="qp-meta-item">Subject: <span>{st.session_state.quiz_subject}</span></div>
        <div class="qp-meta-item">Elapsed: <span>{mins:02d}:{secs:02d}</span></div>
        <div class="qp-meta-item">Score so far: <span>{scored_so_far:.0f} / {total_marks}</span></div>
        <div class="qp-meta-item">This question: <span>{q['marks']} mark{'s' if q['marks']>1 else ''}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

    diff_color = {"easy": "easy", "medium": "medium", "hard": "hard"}[q["difficulty"]]
    marks_map  = {"easy": "1 Mark", "medium": "2 Marks", "hard": "3 Marks"}

    st.markdown(f"""
<div class="q-card">
    <div>
        <span class="q-difficulty {diff_color}">{q['difficulty'].upper()}</span>
        <span class="q-marks">{marks_map[q['difficulty']]}</span>
    </div>
    <div class="q-text">{idx + 1}. {q['text']}</div>
    {'<div class="q-code">' + q.get("code", "") + '</div>' if q.get("code") else ""}
</div>
""", unsafe_allow_html=True)

    if not st.session_state.quiz_answered:
        chosen = st.radio(
            "Select your answer:",
            options=range(len(q["options"])),
            format_func=lambda i: f"{chr(65+i)})  {q['options'][i]}",
            key=f"q_{idx}",
            label_visibility="collapsed"
        )
        st.session_state.quiz_selected = chosen

        btn_label = "Submit Answer" if idx < total - 1 else "Submit & See Results"
        if st.button(btn_label, key="submit_ans"):
            is_correct = (chosen == q["answer"])
            st.session_state.quiz_answers[idx] = {
                "chosen": chosen, "correct": is_correct, "marks": q["marks"] if is_correct else 0
            }
            if is_correct:
                st.session_state.quiz_score += q["marks"]
            st.session_state.quiz_answered = True
            st.rerun()
    else:
        ans_data   = st.session_state.quiz_answers[idx]
        chosen     = ans_data["chosen"]
        is_correct = ans_data["correct"]

        for i, opt in enumerate(q["options"]):
            label = f"{chr(65+i)})  {opt}"
            if i == q["answer"] and i == chosen:
                st.markdown(f'<div style="background:rgba(52,211,153,0.12);border:2px solid #34d399;border-radius:12px;padding:0.75rem 1rem;margin-bottom:8px;color:#34d399;font-weight:600;">✓ {label}</div>', unsafe_allow_html=True)
            elif i == q["answer"]:
                st.markdown(f'<div style="background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.3);border-radius:12px;padding:0.75rem 1rem;margin-bottom:8px;color:#34d399;">✓ {label} (Correct Answer)</div>', unsafe_allow_html=True)
            elif i == chosen:
                st.markdown(f'<div style="background:rgba(248,113,113,0.10);border:1px solid rgba(248,113,113,0.3);border-radius:12px;padding:0.75rem 1rem;margin-bottom:8px;color:#f87171;">✗ {label} (Your Answer)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:0.75rem 1rem;margin-bottom:8px;color:var(--muted);">{label}</div>', unsafe_allow_html=True)

        result_cls = "answer-correct" if is_correct else "answer-wrong"
        result_lbl = "correct" if is_correct else "wrong"
        result_msg = f"✓ Correct! +{q['marks']} mark{'s' if q['marks']>1 else ''}" if is_correct else f"✗ Incorrect. Correct: {chr(65+q['answer'])}) {q['options'][q['answer']]}"
        st.markdown(f"""
<div class="{result_cls}">
    <div class="answer-label {result_lbl}">{result_msg}</div>
    <div class="answer-explain">💡 {q['explanation']}</div>
</div>""", unsafe_allow_html=True)

        if idx < total - 1:
            if st.button("Next Question →", key="next_q"):
                st.session_state.quiz_current  += 1
                st.session_state.quiz_answered  = False
                st.session_state.quiz_selected  = None
                st.rerun()
        else:
            if st.button("🏆  View Results", key="finish_q"):
                st.session_state.quiz_state = "result"
                st.rerun()

# ════════════════════════════════════
# STATE: RESULTS
# ════════════════════════════════════
elif st.session_state.quiz_state == "result":
    questions   = st.session_state.quiz_questions
    score       = st.session_state.quiz_score
    total_marks = sum(q["marks"] for q in questions)
    pct         = (score / total_marks) * 100 if total_marks else 0
    elapsed     = int(time.time() - st.session_state.quiz_start_time) if st.session_state.quiz_start_time else 0
    mins, secs  = divmod(elapsed, 60)

    if pct >= 80:
        result_label, result_color = "Excellent!", "#34d399"
        result_msg = "Outstanding! You're well prepared for campus placements."
    elif pct >= 60:
        result_label, result_color = "Good", "#60a5fa"
        result_msg = "Solid performance! A bit more practice and you'll ace the tests."
    elif pct >= 40:
        result_label, result_color = "Average", "#fbbf24"
        result_msg = "Keep practising — focus on the questions you got wrong."
    else:
        result_label, result_color = "Needs Work", "#f87171"
        result_msg = "Don't get discouraged — review explanations and try again!"

    correct_count   = sum(1 for a in st.session_state.quiz_answers.values() if a["correct"])
    incorrect_count = len(questions) - correct_count

    st.markdown(f"""
<div class="result-card">
    <div style="font-family:'Syne',sans-serif;font-size:0.75rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--muted);margin-bottom:0.5rem;">{st.session_state.quiz_subject} · Quick Test Result</div>
    <div class="result-score" style="color:{result_color};">{score:.0f}<span class="result-max">/{total_marks}</span></div>
    <div class="result-label" style="color:{result_color};">{result_label}</div>
    <div style="font-size:0.9rem;color:var(--muted);margin-bottom:1rem;">{result_msg}</div>
    <div class="result-bar-bg"><div class="result-bar" style="width:{pct:.0f}%;background:linear-gradient(90deg,{result_color},{result_color}88);"></div></div>
    <div style="font-size:0.8rem;color:var(--muted);margin-top:6px;">{pct:.0f}% · Completed in {mins:02d}:{secs:02d}</div>
    <div class="score-breakdown">
        <div class="sb-item"><div class="sb-num" style="color:#34d399;">{correct_count}</div><div class="sb-lbl">Correct</div></div>
        <div class="sb-item"><div class="sb-num" style="color:#f87171;">{incorrect_count}</div><div class="sb-lbl">Incorrect</div></div>
        <div class="sb-item"><div class="sb-num" style="color:#fbbf24;">{score:.0f}</div><div class="sb-lbl">Score / {total_marks}</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1rem;font-weight:700;color:#f0f4ff;margin:1.5rem 0 1rem;display:flex;align-items:center;gap:10px;">Detailed Review <div style="flex:1;height:1px;background:var(--border);margin-left:10px;"></div></div>', unsafe_allow_html=True)

    for i, q in enumerate(questions):
        ans_data    = st.session_state.quiz_answers.get(i, {})
        is_correct  = ans_data.get("correct", False)
        chosen      = ans_data.get("chosen", -1)
        icon        = "✅" if is_correct else "❌"
        bg_color    = "rgba(52,211,153,0.06)" if is_correct else "rgba(248,113,113,0.06)"
        border_col  = "rgba(52,211,153,0.25)" if is_correct else "rgba(248,113,113,0.25)"
        chosen_txt  = q["options"][chosen] if chosen >= 0 else "Not answered"
        correct_txt = q["options"][q["answer"]]

        st.markdown(f"""
<div style="background:{bg_color};border:1px solid {border_col};border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:10px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:0.5rem;">
        <div style="font-size:0.87rem;font-weight:600;color:#e8eeff;line-height:1.5;flex:1;">{icon} Q{i+1}. {q['text'][:120]}{'…' if len(q['text'])>120 else ''}</div>
        <div style="font-size:0.7rem;background:rgba(99,130,255,0.1);border:1px solid rgba(99,130,255,0.2);border-radius:6px;padding:2px 8px;color:#a5b4fc;white-space:nowrap;">{q['marks']} mark{'s' if q['marks']>1 else ''}</div>
    </div>
    <div style="font-size:0.8rem;color:var(--muted);">Your answer: <span style="color:{'#34d399' if is_correct else '#f87171'};">{chosen_txt}</span></div>
    {'<div style="font-size:0.8rem;color:#34d399;margin-top:2px;">Correct: ' + correct_txt + '</div>' if not is_correct else ''}
    <div style="font-size:0.78rem;color:#8895b8;margin-top:6px;line-height:1.5;">💡 {q['explanation'][:220]}{'…' if len(q['explanation'])>220 else ''}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        other = "Coding" if st.session_state.quiz_subject == "Aptitude" else "Aptitude"
        if st.button(f"Try {other} Test →", key="try_other"):
            st.session_state.quiz_state = "select"
            st.rerun()
    with col2:
        if st.button("🔀  Retake (New Shuffle)", key="retake"):
            pool = ALL_APTITUDE if st.session_state.quiz_subject == "Aptitude" else ALL_CODING
            st.session_state.quiz_questions  = pick_questions(pool)
            st.session_state.quiz_state      = "active"
            st.session_state.quiz_current    = 0
            st.session_state.quiz_answers    = {}
            st.session_state.quiz_score      = 0
            st.session_state.quiz_answered   = False
            st.session_state.quiz_selected   = None
            st.session_state.quiz_start_time = time.time()
            st.rerun()