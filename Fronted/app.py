import streamlit as st

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Placement Analyzer",
    page_icon="🎯",
    layout="wide"
)

# ---------------------------
# TITLE
# ---------------------------
st.title("🎯 Placement Readiness & Market Fit Analyzer")
st.markdown("Upload your resume and view your placement insights")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("📌 About Project")
st.sidebar.write("""
This system analyzes student readiness for placement using:
- Resume analysis
- Skill gap identification
- Market fit evaluation
""")

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

# ---------------------------
# MAIN UI
# ---------------------------
if uploaded_file:

    st.success("✅ Resume Uploaded Successfully!")

    st.markdown("---")
    st.subheader("📊 Dashboard")

    # ---------------------------
    # METRICS
    # ---------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎯 Readiness Score", "78%")

    with col2:
        st.metric("📈 Market Fit", "72%")

    with col3:
        st.metric("🛠 Skills Found", "5")

    st.markdown("---")

    # ---------------------------
    # SKILLS SECTION
    # ---------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Extracted Skills")
        st.success("Python")
        st.success("SQL")
        st.success("Communication")
        st.success("Machine Learning")

    with col2:
        st.subheader("❌ Skill Gap")
        st.error("Data Structures")
        st.error("System Design")

    st.markdown("---")

    # ---------------------------
    # MARKET FIT SECTION
    # ---------------------------
    st.subheader("📊 Market Fit Analysis")
    st.info("You have good technical skills but need improvement in Data Structures and real-world projects.")

    # ---------------------------
    # PROGRESS BAR
    # ---------------------------
    st.subheader("📉 Readiness Progress")
    st.progress(78)

    # ---------------------------
    # SUGGESTIONS
    # ---------------------------
    st.subheader("💡 Suggestions")
    st.write("""
    - Practice DSA regularly  
    - Work on real-world projects  
    - Improve communication skills  
    """)

else:
    st.info("👆 Please upload your resume to see analysis")
