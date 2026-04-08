import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = [
    "job_title", "company_name", "location", "industry",
    "employment_type", "job_description", "required_skills",
    "degree_required", "degrees_accepted", "min_salary", "max_salary"
]

# Degree dictionary 
DEGREE_PATTERNS = [
    r"\b(ph\.?d|doctor of philosophy)\b",
    r"\b(m\.?d|doctor of medicine)\b",
    r"\b(m\.?sc|master of science)\b",
    r"\b(b\.?sc|bachelor of science)\b",
    r"\b(m\.?eng|master of engineering)\b",
    r"\b(m\.?a|master of arts)\b",
    r"\b(mba|master of business administration)\b",
    r"\b(bhsc|Bachelor of Health Sciences)\b",
    r"\b(b\.?eng|Bachelor of engineering)\b",
    r"\b(b\.?ed|Bachelor of education)\b",
    r"\b(pgce)\b",
    r"\b(bba|bachelor of business administration)\b",
    r"\b(bcom|bachelor of commerce)\b",
    r"\b(mres|master of research)\b",
    r"\b(mphil|master of philosophy)\b",
    r"\b(b\.?a|bachelor of arts)\b",
    r"\b(b\.?arch|bachelor of architecture)\b",
    r"\b(llb|bachelor of laws)\b",
    r"\b(llm|master of laws)\b",
    r"\b(mbbs|bachelor of medicine)\b",
    r"\b(bds|bachelor of dental surgery)\b",
    r"\b(mpharm|master of pharmacy)\b",
    r"\b(bvetms)\b",
    r"\b(march|master of architecture)\b",
    r"\b(hnd|higher national diploma)\b",
    r"\b(hnc|higher national certificate)\b",
    r"\b(foundation degree)\b",
    r"\b(msci|master of science)\b",
    r"\b(mchem|master of chemistry)\b",
    r"\b(mphys|master of physics)\b",
    r"\b(mmath|master of mathematics)\b",
    r"\b(beng hons|bachelor of engineering with honours)\b",
    r"\bBA Animation\b",
    r"\bBA Business Management\b",
    r"\bBA Criminology\b",
    r"\bBA Culinary Arts\b",
    r"\bBA Education\b",
    r"\bBA English\b",
    r"\bBA Event Management\b",
    r"\bBA Fashion Design\b",
    r"\bBA Film Production\b",
    r"\bBA Game Design\b",
    r"\bBA Graphic Design\b",
    r"\bBA History\b",
    r"\bBA Human Resource Management\b",
    r"\bBA Interior Design\b",
    r"\bBachelors in Marketing\b",
    r"\bBA Media Production\b",
    r"\bBA Media Studies\b",
    r"\bBA Photography\b",
    r"\bBA Politics\b",
    r"\bBA Social Work\b",
    r"\bBArch Architecture\b",
    r"\bBDS Dentistry\b",
    r"\bBEng Aerospace Engineering\b",
    r"\bBEng Automotive Engineering\b",
    r"\bBEng Biomedical Engineering\b",
    r"\bBEng Civil Engineering\b",
    r"\bBEng Electrical Engineering\b",
    r"\bBEng Engineering\b",
    r"\bBEng Environmental Engineering\b",
    r"\bBEng Mechanical Engineering\b",
    r"\bBEng Software Engineering\b",
    r"\bBSc Accounting\b",
    r"\bBSc Actuarial Science\b",
    r"\bBSc Agriculture\b",
    r"\bBSc Animal Science\b",
    r"\bBSc Archaeology\b",
    r"\bBSc Art History\b",
    r"\bBSc Automotive Technology\b",
    r"\bBSc Aviation\b",
    r"\bBSc Biology\b",
    r"\bBSc Biomedical Science\b",
    r"\bBSc Business Information Systems\b",
    r"\bBSc Chemistry\b",
    r"\bBSc Community and Public Service Management\b",
    r"\bBSc Computer Science\b",
    r"\bBSc Construction Management\b",
    r"\bBSc Counselling Psychology\b",
    r"\bBSc Cybersecurity\b",
    r"\bBSc Data Science\b",
    r"\bBSc Dietetics\b",
    r"\bBSc Digital Media\b",
    r"\bBSc Economics\b",
    r"\bBSc Emergency Management\b",
    r"\bBSc Environmental Science\b",
    r"\bBSc Finance\b",
    r"\bBSc Food Science\b",
    r"\bBSc Geography\b",
    r"\bBSc Health Sciences\b",
    r"\bBSc Illustration\b",
    r"\bBSc Information Technology\b",
    r"\bBSc Journalism\b",
    r"\bBSc Logistics and Supply Chain Management\b",
    r"\bBSc Management\b",
    r"\bBSc Mathematics\b",
    r"\bBSc Media and Communication\b",
    r"\bBSc Medical Sciences\b",
    r"\bBSc Midwifery\b",
    r"\bBSc Nursing\b",
    r"\bBSc Occupational Therapy\b",
    r"\bBSc Operations and Supply Chain Management\b",
    r"\bBSc Paramedic Science\b",
    r"\bBSc Physics\b",
    r"\bBSc Physiotherapy\b",
    r"\bBSc Product Design\b",
    r"\bBSc Project Management\b",
    r"\bBSc Psychology\b",
    r"\bBSc Public Policy\b",
    r"\bBSc Real Estate\b",
    r"\bBSc Sociology\b",
    r"\bBSc Sports Science\b",
    r"\bBSc Statistics\b",
    r"\bBSc Supply Chain Management\b",
    r"\bBSc Surveying\b",
    r"\bBSc Textiles\b",
    r"\bBSc User Experience Design\b",
    r"\bBSc Veterinary Medicine and Surgery\b",
    r"\bBSc Visual Communication\b",
    r"\bBVetMS Veterinary Medicine and Surgery\b",
    r"\bBachelor of Library Science\b",
    r"\bHNC Electrical Engineering\b",
    r"\bLLB Law\b",
    r"\bLLM Law\b",
    r"\bLevel 2 Plumbing Diploma\b",
    r"\bMA Criminology\b",
    r"\bMA English\b",
    r"\bMA Film Production\b",
    r"\bMA Graphic Design\b",
    r"\bMA Interior Design\b",
    r"\bMA Marketing\b",
    r"\bMA Media Studies\b",
    r"\bMA Nonprofit Management\b",
    r"\bMA Urban Design\b",
    r"\bMArch Architecture\b",
    r"\bMBA\b",
    r"\bMBBS Medicine\b",
    r"\bMEng Aeronautical Engineering\b",
    r"\bMEng Civil Engineering\b",
    r"\bMEng Electrical Engineering\b",
    r"\bMEng Engineering\b",
    r"\bMEng Mechanical Engineering\b",
    r"\bMEng Software Engineering\b",
    r"\bMEng Structural Engineering\b",
    r"\bMPharm Pharmacy\b",
    r"\bMSc Accounting\b",
    r"\bMSc Animal Science\b",
    r"\bMSc Artificial Intelligence\b",
    r"\bMSc Biology\b",
    r"\bMSc Biomedical Science\b",
    r"\bMSc Building Services\b",
    r"\bMSc Chemistry\b",
    r"\bMSc Clinical Psychology\b",
    r"\bMSc Computer Science\b",
    r"\bMSc Construction Project Management\b",
    r"\bMSc Curriculum Studies\b",
    r"\bMSc Data Science\b",
    r"\bMSc Digital Media\b",
    r"\bMSc Economics\b",
    r"\bMSc Education\b",
    r"\bMSc Educational Leadership\b",
    r"\bMSc Educational Psychology\b",
    r"\bMSc Environmental Science\b",
    r"\bMSc Finance\b",
    r"\bMSc Health Sciences\b",
    r"\bMSc Data Science\b",
    r"\bMSc Hospitality Management\b",
    r"\bMSc Information Technology\b",
    r"\bMSc Management\b",
    r"\bMSc Mathematics\b",
    r"\bMSc Medical Sciences\b",
    r"\bMSc Museum Studies\b",
    r"\bMSc Nursing\b",
    r"\bMSc Nutrition\b",
    r"\bMSc Oral Health Sciences\b",
    r"\bMSc Paediatrics\b",
    r"\bMSc Pharmaceutical Science\b",
    r"\bMSc Primary Education\b",
    r"\bMSc Psychiatry\b",
    r"\bMSc Psychology\b",
    r"\bMSc Radiography\b",
    r"\bMSc Strategic Communication\b",
    r"\bMSc Surgical Sciences\b",
    r"\bMSc Sustainability\b",
    r"\bPhD Biology\b",
    r"\bPhD Computer Science\b",
    r"\bPhD Data Science\b",
    r"\bPhD Economics\b",
    r"\bPhD Environmental Science\b",
    r"\bWater Regulations qualification\b",
]

# Extract degree information and build profile
def extract_degree(text: str) -> str:
    if not text:
        return ""

    # Clean up whitespace but KEEP case for now
    text = re.sub(r'\s+', ' ', text)

    # Define the 'Anchor'
    abbrevs = r"ph\.?d|m\.?sc|m\.?eng|m\.?a|mba|b\.?sc|b\.?eng|b\.?a|llb|mbbs|hnd|hnc"
    
    greedy_pattern = re.compile(
        rf"\b({abbrevs})\b(?:\s+in)?\s+((?:(?!from|at|university|college|managed|worked|using)[A-Z][\w\-]+\s*){{1,4}})",
        re.IGNORECASE
    )

    match = greedy_pattern.search(text)
    if match:
        degree_part = match.group(1).strip()
        subject_part = match.group(2).strip()
        return f"{degree_part} {subject_part}".strip()

    # 4. Fallback: Search your hardcoded list (Longest strings first!)
    sorted_patterns = sorted(DEGREE_PATTERNS, key=len, reverse=True)
    for pattern in sorted_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()

    return ""

# A curated list of common skills to look for in CVs and job descriptions.
SKILLS_DICTIONARY = [
    "python", "java", "sql", "excel", "powerbi", "tableau", "git", "linux",
    "docker", "kubernetes", "aws", "azure", "gcp", "javascript", "html", "css",
    "c", "cpp", "c#", "data analysis", "statistics", "machine learning",
    "deep learning", "nlp", "communication", "teamwork", "problem solving",
    "time management", "customer service", "troubleshooting", "reporting",
    "writing", "research", "organisation", "confidentiality",
    "r", "matlab", "spark", "hadoop", "snowflake", "bigquery", "dbt",
    "looker", "sas", "spss", "etl", "data wrangling",
    "autocad", "solidworks", "catia", "ansys", "revit", "bim",
    "simulink", "fea", "cad", "plc", "scada",
    "patient care", "clinical assessment", "safeguarding", "gdpr",
    "nhs systems", "first aid", "medication management",
    "financial modelling", "bloomberg", "ifrs", "gaap", "vba",
    "forecasting", "budgeting", "risk analysis", "fca",
    "legal research", "contract drafting", "due diligence",
    "litigation", "compliance", "negotiation",
    "agile", "scrum", "prince2", "stakeholder management",
    "change management", "business analysis", "ms project",
    "strategic planning", "risk management", "six sigma", "lean",
    "terraform", "ansible", "jenkins", "ci/cd", "microservices",
    "rest api", "graphql", "nginx", "bash", "Microsoft Office",
    "penetration testing", "siem", "iso 27001", "nist",
    "vulnerability assessment", "incident response", "osint",
    "SEO", "ppc", "google analytics", "hubspot", "salesforce",
    "content strategy", "email marketing", "a/b testing",
    "leadership", "mentoring", "presentation", "stakeholder engagement",
    "critical thinking", "decision making", "conflict resolution","apis", "databases", 
    "project management", "contract law", "data visualization", "pandas",
]

# Build regex patterns once at import time for fast skill matching
SKILL_PATTERNS = []
for s in sorted(SKILLS_DICTIONARY, key=len, reverse=True):
    pattern = r"(?<!\w)" + re.escape(s) + r"(?!\w)"
    SKILL_PATTERNS.append((s, re.compile(pattern)))

DISPLAY_SKILL_MAP = {
    "cpp": "C++",
    "powerbi": "Power BI",
    "nodejs": "Node.js",
    "apis": "APIs",
    "project management": "Project Management",
    "data visualization": "Data Visualization",
    "strategic planning": "Strategic Planning",
    "contract law": "Contract Law",
}


# Text normalization function to clean and standardize text for better matching.
def normalize_text(text: str) -> str:
    """Lowercase, expand abbreviations, remove special chars, compress whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()

    replacements = {
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "js": "javascript",
    }
    for short, full in replacements.items():
        text = re.sub(rf"\b{re.escape(short)}\b", full, text)

    text = text.replace("node.js", "nodejs")
    text = text.replace("power bi", "powerbi")
    text = text.replace("c++", "cpp")

    text = re.sub(r"[^a-z0-9+#/\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords_preserve_short_tokens(text: str) -> str:
    """Remove basic stopwords but keep short tokens like 'c' and 'r' as they can be skills."""
    BASIC_STOPWORDS = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "as", "by", "at", "from", "this", "that", "these", "those", "is", "are",
        "was", "were", "be", "been", "being", "it", "you", "your", "we", "our",
        "they", "their", "will", "would", "should", "can", "may"
    }
    tokens = text.split()
    filtered = [t for t in tokens if (t not in BASIC_STOPWORDS) or (t in {"c", "c#", "c++"})]
    return " ".join(filtered)

# Functions to extract text from uploaded files (PDF, DOCX, TXT)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file using pypdf."""
    from pypdf import PdfReader
    text = []
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def extract_text_from_docx(docx_path: str) -> str:
    """Extract raw text from a DOCX file using python-docx."""
    import docx
    doc = docx.Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(txt_path: str) -> str:
    """Read and return the contents of a plain text file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """Route file to the correct extractor based on its extension."""
    lower = file_path.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    if lower.endswith(".docx"):
        return extract_text_from_docx(file_path)
    if lower.endswith(".txt"):
        return extract_text_from_txt(file_path)
    raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")


def extract_skills(text: str) -> set:
    """Return a set of matched skills from SKILLS_DICTIONARY."""
    text = normalize_text(text)
    found = set()
    for skill, pat in SKILL_PATTERNS:
        if pat.search(text):
            found.add(skill)
    return found

# processing extracted data
def parse_required_skills(cell) -> set:
    """Parse a comma-separated skills string into a normalised set."""
    if pd.isna(cell) or not str(cell).strip():
        return set()
    return {normalize_text(x) for x in str(cell).split(",") if x.strip()}


def parse_degrees_accepted(deg_cell: str) -> list:
    """Parse a semicolon-separated degrees string into a list of individual degrees."""
    if pd.isna(deg_cell):
        return []
    parts = [p.strip() for p in str(deg_cell).split(";")]
    return [p for p in parts if p]


# Normalization
def normalize_degree_text(s: str) -> str:
    """Normalise a degree string for comparison."""
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def degree_match(user_degree: str, degrees_accepted_cell: str) -> bool:
    """Return True if the user's degree matches any entry in degrees_accepted."""
    if not user_degree or not user_degree.strip():
        return False

    user = normalize_degree_text(user_degree)
    accepted = [normalize_degree_text(d) for d in parse_degrees_accepted(degrees_accepted_cell)]

    for a in accepted:
        # Exact match
        if user == a:
            return True
        # User degree is contained in accepted degree
        if len(user) > 3 and user in a:
            return True
    
        user_words = set(user.split())
        accepted_words = set(a.split())
        if len(user_words) >= 2 and user_words.issubset(accepted_words):
            return True

    return False

# Building user profile and job text representations for TF-IDF and skill matching
def build_job_text(row) -> str:
    """Combine relevant job fields into a single normalised text string for TF-IDF."""
    parts = [
        row.get("job_title", ""),
        row.get("company_name", ""),
        row.get("location", ""),
        row.get("industry", ""),
        row.get("employment_type", ""),
        row.get("job_description", ""),
        row.get("required_skills", ""),
        row.get("degree_required", ""),
        row.get("degrees_accepted", ""),
    ]
    joined = " ".join([str(p) for p in parts if not pd.isna(p)])
    joined = normalize_text(joined)
    joined = remove_stopwords_preserve_short_tokens(joined)
    return joined


def build_user_profile_text(degree_text: str = "", cv_text: str = "") -> str:
    """Build a normalised text representation of the user's profile."""
    combined = f"{degree_text} {cv_text}".strip()
    combined = normalize_text(combined)
    combined = remove_stopwords_preserve_short_tokens(combined)
    return combined


def build_user_skills(degree_text: str = "", cv_text: str = "", extra_skills: list = None) -> set:
    """Extract a set of skills from the user's degree, CV text, and any manually entered skills."""
    combined = f"{degree_text} {cv_text}".strip()
    found = extract_skills(combined)
    if extra_skills:
        for skill in extra_skills:
            found.add(normalize_text(skill.strip()))
    return found


def display_skill(s: str) -> str:
    """Return a display-friendly version of an internal skill string."""
    s = str(s)
    return DISPLAY_SKILL_MAP.get(s, s.title())


# Functions to load data, prepare the system, and recommend jobs based on user input.
def load_and_prepare_system(data_path: str):
    """
    Load the dataset, validate columns, build job_skills and job_text,
    fit the TF-IDF vectoriser, and return everything Flask needs at startup.
    Returns: (df, vectorizer, job_tfidf_matrix)
    """
    df = pd.read_csv(data_path, encoding="utf-8")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    if "job_id" not in df.columns:
        df.insert(0, "job_id", range(1, len(df) + 1))

    df["job_skills"] = df.apply(
        lambda r: (
            parse_required_skills(r.get("required_skills", "")) |
            extract_skills(f"{r.get('job_title', '')} {r.get('job_description', '')}")
        ),
        axis=1
    )

    df["job_text"] = df.apply(build_job_text, axis=1)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    job_tfidf = vectorizer.fit_transform(df["job_text"])

    return df, vectorizer, job_tfidf


# The main recommendation function that scores and ranks jobs against the user's profile.
def recommend_jobs(
    df_jobs: pd.DataFrame,
    job_tfidf_matrix,
    vectorizer: TfidfVectorizer,
    degree_text: str = "",
    cv_text: str = "",
    extra_skills: list = None,
    top_n: int = 10,
    w_text: float = 0.55,
    w_skill: float = 0.35,
    w_degree: float = 0.10,
    require_degree_match: bool = False
) -> pd.DataFrame:
    """
    Score and rank jobs against the user's profile.
    final_score = w_text * text_similarity + w_skill * skill_score + w_degree * degree_score
    """
    if not degree_text.strip() and not cv_text.strip() and not extra_skills:
        raise ValueError("Please provide at least a degree, CV text, or skills.")

    if len(degree_text.strip()) < 3 and not cv_text.strip() and not extra_skills:
        raise ValueError("Degree text is too short to produce reliable matches.")

    user_profile = build_user_profile_text(degree_text, cv_text)
    user_skills  = build_user_skills(degree_text, cv_text, extra_skills)

    user_vec  = vectorizer.transform([user_profile])
    text_sims = cosine_similarity(user_vec, job_tfidf_matrix).flatten()

    def compute_skill_row(job_skills):
        if not job_skills:
            return 0.0, [], []
        matched = user_skills & job_skills
        missing = job_skills - user_skills
        score   = len(matched) / max(len(job_skills), 1)
        return score, sorted(matched), sorted(missing)

    skill_results       = df_jobs["job_skills"].apply(compute_skill_row)
    skill_scores        = skill_results.apply(lambda x: x[0]).tolist()
    matched_skills_list = skill_results.apply(lambda x: x[1]).tolist()
    missing_skills_list = skill_results.apply(lambda x: x[2]).tolist()

    def compute_degree_row(row):
        cell = row.get("degrees_accepted") or row.get("degree_required", "")
        ok   = degree_match(degree_text, cell)
        return (1.0 if ok else 0.0, "Yes" if ok else "No")

    degree_results    = df_jobs.apply(compute_degree_row, axis=1)
    degree_scores     = degree_results.apply(lambda x: x[0]).tolist()
    degree_match_list = degree_results.apply(lambda x: x[1]).tolist()

    final_scores = (
        w_text   * text_sims +
        w_skill  * pd.Series(skill_scores).values +
        w_degree * pd.Series(degree_scores).values
    )

    out = df_jobs.copy()
    out["text_similarity"] = text_sims
    out["skill_score"]     = skill_scores
    out["degree_score"]    = degree_scores
    out["final_score"]     = final_scores
    out["degree_match"]    = degree_match_list
    out["matched_skills"]  = matched_skills_list
    out["missing_skills"]  = missing_skills_list

    if require_degree_match and degree_text.strip():
        out = out[out["degree_score"] == 1.0]

    out = out.sort_values("final_score", ascending=False).head(top_n)

    def explain(row):
        ms   = [display_skill(s) for s in row["matched_skills"][:6]]
        miss = [display_skill(s) for s in row["missing_skills"][:6]]
        return (
            f"Degree match: {row['degree_match']} | "
            f"Matched skills: {', '.join(ms) if ms else 'None'} | "
            f"Missing: {', '.join(miss) if miss else 'None'}"
        )

    out["explanation"] = out.apply(explain, axis=1)

    out["salary"] = out.apply(
        lambda r: (
            f"£{int(r['min_salary']):,} - £{int(r['max_salary']):,}"
            if pd.notna(r.get("min_salary")) and pd.notna(r.get("max_salary"))
            else (
                f"£{int(r['avg_salary']):,}"
                if pd.notna(r.get("avg_salary"))
                else "Salary not specified"
            )
        ),
        axis=1
    )

    cols = [
        "job_id", "job_title", "company_name", "location", "industry",
        "employment_type", "experience_level", "salary", "min_salary",
        "max_salary", "avg_salary", "required_skills",
        "job_description", "degree_required", "degrees_accepted",
        "final_score", "text_similarity", "skill_score", "degree_score",
        "degree_match", "matched_skills", "missing_skills", "explanation"
    ]
    return out[cols]

# Function that takes the input and returns the recommended jobs with details for the frontend.  
def recommend_jobs_from_file(
    file_path: str,
    df_jobs: pd.DataFrame,
    job_tfidf_matrix,
    vectorizer: TfidfVectorizer,
    degree_text: str = "",
    extra_skills: list = None,
    top_n: int = 10,
    require_degree_match: bool = False
) -> pd.DataFrame:
    """Extract text from an uploaded CV file and run the recommender."""
    cv_text = extract_text(file_path)
    
    if not degree_text.strip():
        degree_text = extract_degree(cv_text)

    return recommend_jobs(
        df_jobs=df_jobs,
        job_tfidf_matrix=job_tfidf_matrix,
        vectorizer=vectorizer,
        degree_text=degree_text,
        cv_text=cv_text,
        extra_skills=extra_skills,
        top_n=top_n,
        require_degree_match=require_degree_match
    )