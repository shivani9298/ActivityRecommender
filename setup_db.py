import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

conn = psycopg2.connect(
    dbname=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD'),
    host=os.getenv('POSTGRES_HOST'),
    port=os.getenv('POSTGRES_PORT'),
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS activities (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    tags TEXT
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    user_email TEXT,
    activity_title TEXT,
    match_score REAL
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    user_email TEXT,
    activity_title TEXT,
    rating INTEGER,
    keywords TEXT
);
""")

conn.commit()
cursor.close()
conn.close()

print("Database and tables created.")
