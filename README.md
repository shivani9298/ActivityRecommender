# College Plan Generator

This project is an AI-powered web application that generates a personalized college preparation plan for high school students. Based on user input (school, grade, interests, location, dream college, and optional resume), the system uses Google's Gemini API to return a region-specific, actionable plan including course recommendations, extracurriculars, competitions, internships, and a multi-year roadmap.

## Features

- Tailored high school course suggestions (AP/IB/DE)
- Region- and school-specific extracurricular recommendations
- Relevant competitions and honors to target
- Internship and pre-college program suggestions (local and online)
- Accepts resume uploads (PDF) and parses them to improve recommendations
- Automatically adjusts roadmap based on current grade level
- Generates customized multi-year college preparation plans
- Matches student profiles to relevant extracurriculars using semantic embeddings and a machine learning model
- Collects feedback from users on suggested activities
- Feedback-driven retraining: Model is periodically retrained on real user feedback for improved personalization
- Usage logging for observability, analytics, and potential debugging
- Advanced evaluation metrics (RMSE, Pearson correlation, average feedback)

## Technologies Used

- FastAPI — Lightweight, high-performance backend framework.
- PostgreSQL — Database for storing activities, recommendations, and feedback.
- SentenceTransformers — Generates semantic embeddings for activity matching.
- scikit-learn — ML pipeline using `RandomForestRegressor` retrained on user feedback.
- PyPDF2 — Parses uploaded resumes into extractable text.
- Google Generative AI (Gemini API) — Produces region-specific college planning roadmaps.
- Jinja2 Templates — Renders frontend HTML pages.
- Logging & Analytics — Tracks usage and system events in `app.log`.
- dotenv — Safely manages API keys and environment variables.
- pickle — Efficiently stores embedding data and model weights.


## Local Setup

1. Clone the repository:

   git clone https://github.com/shivani9298/ActivityRecommender.git 
   
   cd ActivityRecommender

2. Set up and activate the virtual environment:

   python -m venv venv  
   source venv/bin/activate

3. Install dependencies:

   pip install -r requirements.txt

4. Create a `.env` file and add your keys and database info:

   - GEMINI_API_KEY=your_google_gemini_api_key
   - POSTGRES_DB=your_db_name
   - POSTGRES_USER=your_db_user
   - POSTGRES_PASSWORD=your_db_password
   - POSTGRES_HOST=localhost
   - POSTGRES_PORT=5432

5. Set up the PostgreSQL database:

   python setup_db.py

6. Load the activities dataset and generate embeddings:

   python seed_activities.py  
   python embed_activities.py


7. Run the app locally:

   uvicorn main:app --reload

8. Open your browser to:

   http://127.0.0.1:8000

## Example Workflow

A student uploads their resume and inputs school, grade, interests, and target college. The application then:

- Embeds the combined resume and interest data using a SentenceTransformer model
- Uses a machine learning model (RandomForestRegressor) to score all activities based on semantic similarity and learned user preferences
- Selects and stores the top-matching activities in the PostgreSQL database
- Sends the complete student profile to the Gemini API for contextual plan generation
- Displays both the recommended activities and a personalized multi-year roadmap in the browser
- Collects feedback on recommended activities to refine future suggestions through a feedback-weighted retraining loop
- Calculates and displays advanced model metrics (RMSE, Pearson correlation, average feedback) to monitor and improve recommendation quality


## Future Improvements
- Improve the accuracy of the model by incorporating more high-quality training data.
- Ensure sufficient data variance to enable meaningful Pearson Correlation analysis.



