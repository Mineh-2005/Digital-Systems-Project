import os
import json
import requests
from collections import Counter
from datetime import date
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from NLP import load_and_prepare_system, recommend_jobs, recommend_jobs_from_file, extract_text, extract_degree, extract_skills, display_skill

load_dotenv()

app = Flask(__name__,
            template_folder='system/template',
            static_folder='system/static')

app.secret_key = "marketmatch-secret-key"
CORS(app)

UPLOAD_FOLDER = "system/uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Adzuna config
APP_ID = os.getenv("APP_ID", "")
APP_KEY = os.getenv("APP_KEY", "")
COUNTRY = "gb"

# User memory
users = {}  # { email: { name, password } }

# Admin access
ADMIN_KEY = os.getenv("ADMIN_KEY", "marketmatch-admin-2026")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@marketmatch.com")

# Load dataset and build TF-IDF model once at startup
print("Loading dataset and preparing recommendation system...")
df_jobs, vectorizer, job_tfidf = load_and_prepare_system("dataset.csv")

with open("dataset_version.json", "r") as f:
    DATASET_VERSION = json.load(f)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_adzuna_jobs(search_term="software engineer", results_per_page=20, page=1):
    """
    Fetch live jobs from Adzuna and return them in a frontend-friendly format.
    """
    if not APP_ID or not APP_KEY:
        raise ValueError("Adzuna API credentials are missing. Set APP_ID and APP_KEY.")

    url = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search/{page}"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": results_per_page,
        "what": search_term,
        "content-type": "application/json"
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    formatted_jobs = []

    for job in results:
        salary_value = (
            job.get("salary_max")
            or job.get("salary_min")
            or 0
        )

        formatted_jobs.append({
            "job_id": job.get("id"),
            "job_title": job.get("title", "Untitled role"),
            "company_name": job.get("company", {}).get("display_name", "Unknown company"),
            "location": job.get("location", {}).get("display_name", "Unknown location"),
            "industry": "Live market role",
            "employment_type": job.get("contract_type", "Not specified"),
            "experience_level": job.get("contract_time", "Not specified"),
            "annual_salary": int(salary_value) if salary_value else 0,
            "job_description": job.get("description", ""),
            "redirect_url": job.get("redirect_url", ""),
            "matched_skills": [],
            "missing_skills": [],
            "degree_match": "Unknown",
            "final_score": 0.0
        })

    return {
        "count": data.get("count", 0),
        "results": formatted_jobs,
        "raw_results": results
    }


# Page routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/saved-jobs")
def saved_jobs():
    return render_template("saved-jobs.html")

@app.route("/profile")
def profile():
    return render_template("profile.html")

@app.route("/job-details")
def job_details():
    return render_template("job-details.html")

@app.route("/forgot-password")
def forgot_password():
    return render_template("forgot-password.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/admin")
def admin_page():
    return render_template("admin.html")

# API routes

@app.route("/api/signup", methods=["POST"])
def api_signup():
    """Register a new user."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data received."}), 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not name or not email or not password:
        return jsonify({"success": False, "error": "Name, email and password are required."}), 400
    if len(password) < 8:
        return jsonify({"success": False, "error": "Password must be at least 8 characters."}), 400

    if email in users:
        return jsonify({"success": False, "error": "An account with this email already exists."}), 400
    users[email] = {"name": name, "password": password}
    session["user"] = {"name": name, "email": email}
    return jsonify({"success": True, "name": name, "email": email})


@app.route("/api/login", methods=["POST"])
def api_login():
    """Authenticate a user."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data received."}), 400

    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required."}), 400
    
    is_admin = email.lower() == ADMIN_EMAIL.lower()

    if is_admin:
        name = "Admin"
    else:
        user = users.get(email)
        if not user or user["password"] != password:
            return jsonify({"success": False, "error": "Invalid email or password."}), 401
        name = user["name"]
    session["user"] = {"name": name, "email": email, "is_admin": is_admin}
    return jsonify({"success": True, "name": name, "email": email, "is_admin": is_admin})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """Log the user out."""
    session.clear()
    return jsonify({"success": True})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    try:
        degree_text = request.form.get("degree", "").strip()
        skills_input = request.form.get("skills", "").strip()
        extra_skills = [s.strip() for s in skills_input.split(",") if s.strip()] if skills_input else []
        file = request.files.get("cv_file")

        used_file = False 

        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Extract text, degree and skills from CV BEFORE deleting
            cv_raw_text = extract_text(file_path)
            cv_extracted_degree = extract_degree(cv_raw_text)
            cv_extracted_skills = extract_skills(cv_raw_text)

            results = recommend_jobs_from_file(
                file_path=file_path,
                df_jobs=df_jobs,
                job_tfidf_matrix=job_tfidf,
                vectorizer=vectorizer,
                degree_text=degree_text or cv_extracted_degree,  # use extracted if none typed
                extra_skills=extra_skills,
                top_n=10,
                require_degree_match=False
            )

            os.remove(file_path)
            used_file = True

        else:
            results = recommend_jobs(
                df_jobs=df_jobs,
                job_tfidf_matrix=job_tfidf,
                vectorizer=vectorizer,
                degree_text=degree_text,
                cv_text="",
                extra_skills=extra_skills,
                top_n=10,
                require_degree_match=False
            )

        records = results.to_dict(orient="records")
        for r in records:
            if isinstance(r.get("matched_skills"), (set, list)):
                r["matched_skills"] = list(r["matched_skills"])
            if isinstance(r.get("missing_skills"), (set, list)):
                r["missing_skills"] = list(r["missing_skills"])

        if used_file:
            extracted_degree = degree_text or cv_extracted_degree or ""
            extracted_skills = [display_skill(s) for s in cv_extracted_skills][:10]
        else:
            extracted_degree = degree_text or ""
            extracted_skills = extra_skills

        return jsonify({
            "success": True,
            "results": records,
            "extracted_degree": extracted_degree,
            "extracted_skills": extracted_skills
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/job/<int:job_id>", methods=["GET"])
def api_job_detail(job_id):
    """Return a single job by ID."""
    job = df_jobs[df_jobs["job_id"] == job_id]
    if job.empty:
        return jsonify({"success": False, "error": "Job not found."}), 404
    return jsonify({"success": True, "job": job.iloc[0].to_dict()})


@app.route("/api/profile", methods=["GET"])
def api_get_profile():
    """Return the current user's profile from session."""
    user = session.get("user")
    if not user:
        return jsonify({"success": False, "error": "Not logged in."}), 401
    return jsonify({"success": True, "profile": user})

@app.route("/api/profile", methods=["POST"])
def api_update_profile():
    """Update the current user's profile."""
    if "user" not in session:
        return jsonify({"success": False, "error": "Not logged in."}), 401

    data = request.get_json()
    
    # Update the session dictionary
    if "name" in data:
        session["user"]["name"] = data["name"]
    if "email" in data:
        session["user"]["email"] = data["email"]
    
    # Mark session as modified to ensure Flask saves it
    session.modified = True 

    return jsonify({"success": True, "profile": session["user"]})


@app.route("/api/live-jobs")
def live_jobs():
    """
    Live jobs endpoint for Apply Now buttons.
    Returns redirect_url for real job applications.
    """
    try:
        search_term = request.args.get("q", "software engineer").strip()
        live_data = get_adzuna_jobs(search_term=search_term, results_per_page=20, page=1)

        return jsonify({
            "success": True,
            "results": live_data["results"],
            "count": live_data["count"]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route("/api/admin/upload-dataset", methods=["POST"])
def admin_upload_dataset():
    """Allow admin to upload a new dataset CSV."""
    global df_jobs, vectorizer, job_tfidf, DATASET_VERSION

    # Check admin key
    user = session.get("user")
    if not user or not user.get("is_admin"):
        return jsonify({"success": False, "error": "Unauthorised."}), 401

    file = request.files.get("dataset")
    if not file or not file.filename.endswith(".csv"):
        return jsonify({"success": False, "error": "Please upload a CSV file."}), 400

    try:
        # Save new dataset
        file.save("dataset.csv")

        # Reload the system with new data
        df_jobs, vectorizer, job_tfidf = load_and_prepare_system("dataset.csv")

        with open("dataset_version.json", "r") as f:
            version_data = json.load(f)

        # Bump version number
        parts = version_data["version"].split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        version_data["version"]      = ".".join(parts)
        version_data["last_updated"] = str(date.today())
        version_data["records"]      = len(df_jobs)
        version_data["changes"].append(f"Dataset updated on {date.today()} — {len(df_jobs)} records")

        with open("dataset_version.json", "w") as f:
            json.dump(version_data, f, indent=4)
        DATASET_VERSION = version_data 

        return jsonify({
            "success": True,
            "message": f"Dataset updated successfully. {len(df_jobs)} jobs loaded.",
            "version": version_data["version"],
            "records": len(df_jobs)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/dataset-version", methods=["GET"])
def api_dataset_version():
    """Return current dataset version info."""
    return jsonify({
        "success": True,
        "version": DATASET_VERSION
    })

@app.route("/api/forgot-password", methods=["POST"])
def api_forgot_password():
    data = request.get_json()
    email = data.get("email", "").strip()
    if not email:
        return jsonify({"success": False, "error": "Email is required."}), 400
    return jsonify({"success": True})

@app.route("/api/change-password", methods=["POST"])
def api_change_password():
    user = session.get("user")
    if not user:
        return jsonify({"success": False, "error": "Not logged in."}), 401

    data = request.get_json()
    current = data.get("current_password", "")
    new_pw  = data.get("new_password", "")

    email = user.get("email")
    stored = users.get(email)
    if not stored or stored["password"] != current:
        return jsonify({"success": False, "error": "Current password is incorrect."}), 401
    if len(new_pw) < 8:
        return jsonify({"success": False, "error": "Password must be at least 8 characters."}), 400

    users[email]["password"] = new_pw
    return jsonify({"success": True})

@app.route("/api/delete-account", methods=["POST"])
def api_delete_account():
    user = session.get("user")
    if not user:
        return jsonify({"success": False, "error": "Not logged in."}), 401
    email = user.get("email")
    users.pop(email, None)
    session.clear()
    return jsonify({"success": True})

@app.route("/api/market", methods=["GET"])
def api_market():
    """
    Live market data for prediction section.
    Uses Adzuna for current count, salary and a simple 6-month forecast.
    """
    try:
        degree = request.args.get("degree", "").strip()

        search_term = degree if degree else "software engineer"
        live_data = get_adzuna_jobs(search_term=search_term, results_per_page=50, page=1)

        raw_jobs = live_data["raw_results"]
        current_count = live_data["count"]

        if not raw_jobs:
            return jsonify({"success": False, "error": "No market data found."}), 404

        salary_values = []
        all_skills = []
        industry_counter = Counter()
        location_counter = Counter()

        for job in raw_jobs:
            salary = job.get("salary_max") or job.get("salary_min")
            if salary:
                salary_values.append(int(salary))

            category = job.get("category", {}).get("label")
            if category:
                industry_counter[category] += 1

            location = job.get("location", {}).get("display_name")
            if location:
                location_counter[location] += 1

            description = job.get("description", "")
            for token in ["Python", "JavaScript", "SQL", "React", "AWS", "Docker", "Java", "Git", "Excel", "Power BI"]:
                if token.lower() in description.lower():
                    all_skills.append(token)

        avg_salary = int(sum(salary_values) / len(salary_values)) if salary_values else 0
        min_salary = min(salary_values) if salary_values else 0
        max_salary = max(salary_values) if salary_values else 0

        skill_counts = Counter(all_skills)
        top_skills = [
            {"skill": skill, "demand": round((count / len(raw_jobs)) * 100, 1)}
            for skill, count in skill_counts.most_common(8)
        ]

        jobs_by_industry = dict(industry_counter.most_common(8))
        top_locations = dict(location_counter.most_common(6))

        # Baseline job counts per category to determine growth direction
        # Higher count = more in-demand field = positive growth
        # Lower count = declining field = negative growth
        BASELINE_COUNT = 50000  # average UK job count across all fields

        # thresholds first
        growth_rate = 0.0

        if current_count >= 100000:
            growth_rate = 18.0   # very high demand e.g. software, nursing
        elif current_count >= 50000:
            growth_rate = 10.0   # healthy demand
        elif current_count >= 20000:
            growth_rate = 4.0    # stable
        elif current_count >= 5000:
            growth_rate = -3.0   # slow decline
        elif current_count >= 1000:
            growth_rate = -8.0   # declining field
        else:
            growth_rate = -15.0  # very low demand e.g. carpenter, typist

        
        forecast_labels = ["Now"] + [f"Month {i}" for i in range(1, 7)]
        forecast_points = [current_count]

        monthly_growth = growth_rate / 6

        for i in range(1, 7):
            next_value = int(forecast_points[i - 1] * (1 + monthly_growth / 100))
            forecast_points.append(next_value)
            
        return jsonify({
            "success": True,
            "degree_filter": degree or "All degrees",
            "total_jobs": current_count,
            "current_count": current_count,
            "avg_salary": avg_salary,
            "min_salary": min_salary,
            "max_salary": max_salary,
            "growth_rate": growth_rate,       
            "top_skills": top_skills,
            "jobs_by_industry": jobs_by_industry,
            "top_locations": top_locations,
            "forecast_labels": forecast_labels,
            "forecast_points": forecast_points
            
        })
    
    except Exception as e:
        print("ERROR IN /api/market:", e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)