import fitz          # PyMuPDF
import json
import re
from groq import Groq

client = Groq(api_key="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")


# ── TEXT EXTRACTION ─────────────────────────────────────────────────────────
def extract_resume_text(file_path: str) -> str:
    """Extract plain text from PDF or DOCX."""
    ext = file_path.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text("text")
        except Exception as e:
            print(f"PDF read error: {e}")
        return text.strip()

    elif ext == "docx":
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except ImportError:
            # python-docx not installed — try raw extraction
            try:
                import zipfile, xml.etree.ElementTree as ET
                with zipfile.ZipFile(file_path) as z:
                    xml_content = z.read("word/document.xml")
                root = ET.fromstring(xml_content)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                texts = [node.text or "" for node in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")]
                return " ".join(texts).strip()
            except Exception as e:
                print(f"DOCX read error: {e}")
                return ""
    return ""


# ── REGEX FALLBACK ───────────────────────────────────────────────────────────
_COMMON_TECH_SKILLS = {
    "python","java","javascript","typescript","c++","c","c#","go","rust","kotlin","swift",
    "sql","mysql","postgresql","mongodb","sqlite","nosql","firebase",
    "html","css","react","angular","vue","nextjs","nodejs","express","django","flask","fastapi","spring",
    "aws","azure","gcp","docker","kubernetes","terraform","linux","git","github","gitlab","ci/cd",
    "machine learning","deep learning","neural network","nlp","computer vision",
    "tensorflow","pytorch","keras","scikit-learn","sklearn","xgboost","pandas","numpy","matplotlib","seaborn","plotly",
    "tableau","power bi","excel","spark","hadoop","kafka","airflow",
    "rest api","graphql","microservices","agile","scrum","data structures","algorithms","oop",
    "opencv","langchain","llm","transformer","bert","gpt",
}

def _regex_fallback(text: str) -> dict:
    """Best-effort extraction using regex when AI fails."""
    lower = text.lower()

    # Skills
    found_skills = []
    for skill in _COMMON_TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, lower):
            # Preserve original capitalisation from common mapping
            display = {s.lower(): s for s in [
                "Python","Java","JavaScript","TypeScript","C++","C#","Go","Rust","Kotlin","Swift",
                "SQL","MySQL","PostgreSQL","MongoDB","SQLite","NoSQL","Firebase",
                "HTML","CSS","React","Angular","Vue","Next.js","Node.js","Express","Django","Flask",
                "FastAPI","Spring","AWS","Azure","GCP","Docker","Kubernetes","Terraform","Linux",
                "Git","GitHub","GitLab","CI/CD","Machine Learning","Deep Learning","NLP",
                "TensorFlow","PyTorch","Keras","scikit-learn","XGBoost","Pandas","NumPy",
                "Matplotlib","Seaborn","Plotly","Tableau","Power BI","Excel","Spark","Hadoop",
                "Kafka","Airflow","REST API","GraphQL","Microservices","Agile","Scrum",
                "Data Structures","Algorithms","OOP","OpenCV","LangChain","LLM","Transformer",
                "BERT","GPT", "Computer Vision",
            ]}.get(skill, skill.title())
            found_skills.append(display)

    # Projects — count headings / bullet sections mentioning "project"
    project_count = len(re.findall(
        r'(?i)(project\s*[:\-–|]|\bprojects?\b\s*\n)',
        text
    ))
    # Also count lines that look like project titles after a Projects heading
    proj_section = re.search(r'(?i)projects?\s*[\n:](.*?)(?:\n[A-Z][^\n]{0,40}\n|\Z)', text, re.DOTALL)
    if proj_section:
        bullets = re.findall(r'\n\s*[-•*]\s+\S', proj_section.group(1))
        project_count = max(project_count, len(bullets))
    project_count = min(max(project_count, 0), 15)

    # Internships
    internship_count = len(re.findall(
        r'(?i)(intern(ship)?|trainee|apprentice)',
        text
    ))
    # Each mention ≈ one role; cap at 5
    internship_count = min(internship_count, 5)

    # Certifications
    cert_count = len(re.findall(
        r'(?i)(certif(ied|ication|icate)|credential|badge|course completion)',
        text
    ))
    cert_count = min(cert_count, 10)

    return {
        "projects": project_count,
        "internships": internship_count,
        "certifications": cert_count,
        "skills": found_skills[:20]
    }


# ── AI PARSER ────────────────────────────────────────────────────────────────
def parse_resume_with_ai(resume_text: str) -> dict:
    """
    Try AI parsing first; fall back to regex if AI fails or returns bad JSON.
    """
    if not resume_text or len(resume_text) < 50:
        return {"projects": 0, "internships": 0, "certifications": 0, "skills": []}

    prompt = f"""You are a resume parser. Extract data from the resume below.

Return ONLY a JSON object — no markdown, no explanation, nothing else.

JSON format:
{{
  "projects": <integer count of distinct projects>,
  "internships": <integer count of distinct internships or work experiences>,
  "certifications": <integer count of certifications or courses completed>,
  "skills": [<list of technical skill strings>]
}}

Rules:
- Count ONLY items clearly present in the resume.
- skills: extract ALL technical tools, languages, frameworks, and libraries mentioned.
- If a section is missing, use 0 or [].
- Output ONLY valid JSON.

RESUME:
{resume_text[:4000]}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a resume parser. Output only valid JSON, nothing else."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0,
            max_tokens=800
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")

        # Extract JSON object
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group()

        parsed = json.loads(raw)

        # Validate required keys exist
        result = {
            "projects":       int(parsed.get("projects", 0)),
            "internships":    int(parsed.get("internships", 0)),
            "certifications": int(parsed.get("certifications", 0)),
            "skills":         list(parsed.get("skills", []))
        }

        # If AI returned empty skills but text is substantial, top-up with regex
        if len(result["skills"]) == 0 and len(resume_text) > 100:
            fallback = _regex_fallback(resume_text)
            result["skills"] = fallback["skills"]
            if result["projects"] == 0:
                result["projects"] = fallback["projects"]
            if result["internships"] == 0:
                result["internships"] = fallback["internships"]
            if result["certifications"] == 0:
                result["certifications"] = fallback["certifications"]

        return result

    except Exception as e:
        print(f"AI parse failed ({e}), using regex fallback.")
        return _regex_fallback(resume_text)


# ── CLI TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_resume.pdf"
    text = extract_resume_text(path)
    print(f"Extracted {len(text)} characters.")
    result = parse_resume_with_ai(text)
    print(json.dumps(result, indent=2))