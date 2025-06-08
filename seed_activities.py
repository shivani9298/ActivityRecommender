import psycopg2
import os
from dotenv import load_dotenv
import csv

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD'),
    host=os.getenv('POSTGRES_HOST', 'localhost'),
    port=os.getenv('POSTGRES_PORT', '5432')
)
cursor = conn.cursor()

with open("activities.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        cursor.execute("""
            INSERT INTO activities (title, type, description, tags)
            VALUES (%s, %s, %s, %s)
        """, (row['title'], row['type'], row['description'], row['tags']))

conn.commit()
conn.close()

print("Activities seeded.")
