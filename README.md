**Student:** Mineh Issa Gholian  
**Student Number:** 24018724  
**Academic Year:** 2025-26
**Module:** UFCFXK-30-3 Digital Systems Project 

# MarketMatch — AI-Driven Job Market Intelligence System

An AI-powered web application that recommends personalised job opportunities to students and graduates based on their uploaded CV or manually entered degree and skills profile. Built as a final year dissertation project at the University of the West of England, Bristol.

---

## Project Overview

MarketMatch addresses the cold start problem that affects graduates on platforms like LinkedIn and Indeed, where recommendations rely on historical user behaviour. Instead, MarketMatch uses a content-based NLP pipeline to match CV content and degree qualifications directly against a curated dataset of 1,000 UK job postings, producing explainable, ranked recommendations without requiring any prior user history.

The system integrates the Adzuna Jobs API to provide live market intelligence including vacancy counts, salary benchmarks, and field-specific demand forecasts.

---

## Features

- CV upload support for PDF, DOCX, and TXT file formats
- Manual profile entry via degree and skills input fields
- NLP-powered skill extraction and text normalisation
- Hybrid weighted recommendation algorithm (TF-IDF cosine similarity + skill matching + degree eligibility)
- Explainable match scores showing matched skills, missing skills, and score breakdowns
- Results filtering by location, salary, employment type, and experience level
- Live market intelligence panel with vacancy trends powered by the Adzuna API
- Save jobs for later review
- Admin panel for dataset management and versioning
- User authentication with session management

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.12.3, Flask 3.1.3 |
| Algorithm | scikit-learn 1.8.0 (TF-IDF, cosine similarity), pandas 3.0.1 |
| CV Parsing | pypdf 6.9.2, python-docx 1.2.0 |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| External API | Adzuna Jobs API |
| Styling | Custom CSS design system with CSS variables |

---

## Project Structure

```
Final Year Project/
│
├── app.py                      # Flask application and all API routes
├── NLP.py                      # NLP pipeline and recommendation algorithm
├── dataset.csv                 # Original dataset (1000 rows, 35 industries)
├── dataset 2.csv               # For test admin page (updating dataset)  
├── dataset_version.json        # Dataset versioning metadata
├── README.MD                   # README file with instructions
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (.en.example in repo)
│
├── system/
│   ├── template/               # HTML pages (Flask templates)
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── signup.html
│   │   ├── upload.html
│   │   ├── results.html
│   │   ├── job-details.html
│   │   ├── saved-jobs.html
│   │   ├── dashboard.html
│   │   ├── profile.html
│   │   ├── admin.html
│   │   ├── forgot-password.html
│   │   ├── privacy.html
│   │   └── terms.html
│   │
│   ├── static/                 # Static assets
│   │   ├── styles.css
│   │   └── script.js
│   │
│   └── uploads/                # Temporary storage for uploaded CV files
│
│
│   # Sample CVs for testing file input
│
├── sample_cv_law.docx  
├── sample_cv_data_analyst.pdf  
├── sample_cv_mechanical_engineering.txt  
├── sample_cv_medical.pdf 
├── sample_cv_marketing.docx  
│
├── System Architecture and Data Flow Diagram.jpeg
│
└── .venv/                      # Python virtual environment (not included in repo)
```

---

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- An Adzuna API account (free tier available at https://developer.adzuna.com)

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd "Final Year Project"
```

### 2. Create and activate a virtual environment

On Windows:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root with the following contents:

```
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key
ADMIN_EMAIL=admin@marketmatch.com
ADMIN_KEY=marketmatch-admin-2026
ADMIN password= 12345678
```

To obtain Adzuna API credentials, register for a free account at https://developer.adzuna.com and create an application to receive your App ID and App Key.

### 5. Run the application

```bash
python app.py
```

The application will start on http://127.0.0.1:5000. Open this URL in your browser to access MarketMatch.

You should see the following output in your terminal when the server starts successfully:

```
Loading dataset and preparing recommendation system...
Ready — 1000 jobs loaded.
 * Running on http://127.0.0.1:5000
```

---

## Usage

### Standard user flow

1. Navigate to http://127.0.0.1:5000 and create an account via the Sign Up page.
2. On the Upload page, either upload a CV file (PDF, DOCX, or TXT) or enter your degree and skills manually.
3. Click Find Matching Jobs to run the recommendation algorithm.
4. View your ranked job matches on the Results page, with filters for location, salary, employment type, and experience level.
5. Click View Details on any job to see the full match score breakdown, matched skills, and missing skills.
6. Save jobs for later review via the Saved Jobs page.
7. Return to the Dashboard to view your profile summary and session statistics.

### Admin access

Log in using the email address defined in the `ADMIN_EMAIL` environment variable. An Admin link will appear in the navigation bar, providing access to the dataset management panel where new CSV datasets can be uploaded to replace the current one.

---

## Algorithm

The recommendation algorithm is a hybrid weighted scoring system combining three components:

```
final_score = (0.55 x text_similarity) + (0.35 x skill_score) + (0.10 x degree_score)
```

- **text_similarity**: Cosine similarity between the TF-IDF vector of the user's profile and each job's combined text fields.
- **skill_score**: The ratio of skills matched between the user's profile and the job's required skills list.
- **degree_score**: A binary score (1.0 if the user's degree appears in the job's accepted degrees list, 0.0 otherwise).

The NLP pipeline processes input text through five stages: normalisation, stopword removal, skill extraction, TF-IDF vectorisation (unigrams and bi-grams), and weighted scoring. The model is built once at server startup and reused across all recommendation requests.

---

## Dependencies

Full dependency list as defined in requirements.txt:

```
blinker==1.9.0
certifi==2026.2.25
charset-normalizer==3.4.6
click==8.3.1
colorama==0.4.6
Flask==3.1.3
flask-cors==6.0.2
idna==3.11
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.5.3
lxml==6.0.2
MarkupSafe==3.0.3
numpy==2.4.3
pandas==3.0.1
pypdf==6.9.2
PyPDF2==3.0.1
python-dateutil==2.9.0.post0
python-docx==1.2.0
python-dotenv==1.2.2
requests==2.33.0
scikit-learn==1.8.0
scipy==1.17.1
six==1.17.0
threadpoolctl==3.6.0
typing_extensions==4.15.0
tzdata==2025.3
urllib3==2.6.3
Werkzeug==3.1.7
```

---

## Known Limitations

- Job descriptions in the dataset were generated synthetically using templates, which reduces the discriminability of the TF-IDF component for semantically distinct industries.
- User authentication uses Flask sessions and browser localStorage. There is no persistent database; session data is cleared on server restart.
- The system is designed to run locally. It is not configured for production deployment.
- The skills dictionary, while covering over 80 skills across multiple domains, is hand-curated and may not capture all emerging or specialised skills.

---

## Academic Context

This project was developed as a final year dissertation for the module Digital Systems Project (UFCFXK-30-3) at the University of the West of England, Bristol, under the supervision of Dr. Elias Pimenidis.

---

## Licence

This project was developed for academic purposes. All rights reserved by the author.
