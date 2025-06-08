from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import logging
from collections import defaultdict
import psycopg2
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# Logging setup
usage_counter = defaultdict(int)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    filename="app.log",
    filemode="a"
)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load embedding model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
with open("activity_embeddings.pkl", "rb") as f:
    activity_rows, activity_embeddings = pickle.load(f)

# Load trained ML model
with open("ml_model.pkl", "rb") as f:
    trained_model = pickle.load(f)

# Helper to get PostgreSQL connection
def get_pg_conn():
    return psycopg2.connect(
        dbname=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432')
    )

def extract_bullets(plan_text):
    return re.findall(r"- (.+)", plan_text)

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/generate-plan", response_class=HTMLResponse)
async def generate_plan(
    request: Request,
    user_email: str = Form(...),
    school: str = Form(...),
    grade: str = Form(...),
    interests: str = Form(...),
    state: str = Form(...),
    dream_college: str = Form(...),
    resume: UploadFile = File(None)
):
    logging.info(f"Received plan request from: {user_email}")
    usage_counter[user_email] += 1
    logging.info(f"{user_email} has used the app {usage_counter[user_email]} times")

    resume_text = ""
    if resume:
        try:
            with open("temp_resume.pdf", "wb") as f:
                f.write(await resume.read())
            with open("temp_resume.pdf", "rb") as f:
                reader = PdfReader(f)
                resume_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            resume_text = f"(Error reading resume: {e})"

    user_text = f"Resume: {resume_text}\nInterests: {interests}\nDream college: {dream_college}"
    user_embedding = model.encode([user_text])[0]

    # Evaluate cosine similarity 3 times and average
    cosine_scores = []
    for _ in range(3):
        cosine_scores.append(cosine_similarity([user_embedding], activity_embeddings)[0])

    # Use ML model to score activities
    scores = trained_model.predict(activity_embeddings)
    # Get unique top recommendations 
    seen_titles = set()
    top_indices = []
    for idx in np.argsort(scores)[::-1]:
        title = activity_rows[idx][1]
        if title not in seen_titles:
            seen_titles.add(title)
            top_indices.append(idx)
        if len(top_indices) == 5:
            break
    top_recs = [activity_rows[i] for i in top_indices]

    conn = get_pg_conn()
    cursor = conn.cursor()

    for (activity_id, title, desc), score in zip(top_recs, scores[top_indices]):
        cursor.execute("""
            INSERT INTO recommendations (user_email, activity_title, match_score)
            VALUES (%s, %s, %s)
        """, (user_email, title, float(score)))

    conn.commit()

    # Calculate RMSE and Pearson correlation
    cursor.execute("""
        SELECT rating, activity_title FROM feedback
    """)
    feedback_rows = cursor.fetchall()
    if feedback_rows and len(feedback_rows) > 1:
        y_true = [row[0] for row in feedback_rows]
        y_pred = []
        for _, activity_title in feedback_rows:
            idx = next((i for i, row in enumerate(activity_rows) if row[1] == activity_title), None)
            if idx is not None:
                y_pred.append(float(scores[idx]))
            else:
                y_pred.append(0.0)
        # RMSE 
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_metric = f"RMSE: {rmse:.2f}"
        # Pearson correlation
        try:
            pearson_corr, _ = pearsonr(y_true, y_pred)
            pearson_metric = f"Pearson Corr: {pearson_corr:.2f}"
        except Exception:
            pearson_metric = "Pearson Corr: N/A"
    else:
        rmse_metric = "RMSE: N/A (not enough feedback)"
        pearson_metric = "Pearson Corr: N/A (not enough feedback)"
    feedback_avg = np.mean([row[0] for row in feedback_rows]) if feedback_rows else None
    accuracy_metric = feedback_avg if feedback_avg is not None else 'N/A'

    conn.close()

    recommendation_list = "\n".join([
        f"- {title} (score: {score:.2f})" for (_, title, _), score in zip(top_recs, scores[top_indices])
    ])

    try:
        gemini_prompt = f"""
Student at {school}, grade {grade}, located in {state}, interested in {interests}, wants to attend {dream_college}.

Resume context (if any):
{resume_text}

Please generate a personalized college preparation plan using the format below. Include real, 
regionally specific programs, clubs, competitions, and internships relevant to my location and high school. 
Do not include introductions, markdown, emojis, or extra formatting. Output in clean plain text. (Do not 
just put two bullet points but maintain this format)

High School Courses:
- [course 1]
- [course 2]

Relevant Extracurriculars:
In School:
- [activity 1]
- [activity 2]

Outside of School:
- [activity 3]
- [activity 4]

Competitions/Honors:
- [competition 1]
- [competition 2]

Programs/Internships:
- [program 1]
- [program 2]

College Preparation Roadmap (starting grade {grade}):
Year 1:
- [milestone 1]
- [milestone 2]

Year 2:
- [milestone 3]
- [milestone 4]

{ "Year 3:\n- [milestone 5]\n- [milestone 6]" if grade in ["9", "10"] else "" }
{ "Year 4:\n- [milestone 7]\n- [milestone 8]" if grade == "9" else "" }
"""
        gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
        response = gemini_model.generate_content(gemini_prompt)
        plan = response.text
    except Exception as e:
        plan = f"(Error generating plan: {e})"

    return templates.TemplateResponse("form.html", {
        "request": request,
        "plan": plan,
        "recommendations": recommendation_list,
        "user_email": user_email,
        "accuracy_metric": accuracy_metric,
        "rmse_metric": rmse_metric,
        "pearson_metric": pearson_metric
    })

@app.post("/submit-feedback", response_class=HTMLResponse)
async def submit_feedback(
    request: Request,
    rating: int = Form(...),
    user_email: str = Form(...),
    activity_title: str = Form(...)
):
    conn = get_pg_conn()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO feedback (user_email, activity_title, rating)
        VALUES (%s, %s, %s)
    """, (user_email, activity_title, rating))

    conn.commit()
    conn.close()

    return templates.TemplateResponse("form.html", {
        "request": request,
        "plan": None,
        "message": "Thank you for your feedback!"
    })
