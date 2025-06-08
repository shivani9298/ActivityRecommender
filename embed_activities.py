from sentence_transformers import SentenceTransformer
import psycopg2
import os
from dotenv import load_dotenv
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

conn = psycopg2.connect(
    dbname=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD'),
    host=os.getenv('POSTGRES_HOST', 'localhost'),
    port=os.getenv('POSTGRES_PORT', '5432')
)
cursor = conn.cursor()

cursor.execute("SELECT id, title, description FROM activities")
rows = cursor.fetchall()

texts = [f"{title}. {desc}" for _, title, desc in rows]
embeddings = model.encode(texts)

with open("activity_embeddings.pkl", "wb") as f:
    pickle.dump((rows, embeddings), f)

print("Embedded all activities.")

# Load activity embeddings and rows
with open("activity_embeddings.pkl", "rb") as f:
    activity_rows, activity_embeddings = pickle.load(f)

# Map activity titles to embeddings
title_to_embedding = {row[1]: emb for row, emb in zip(activity_rows, activity_embeddings)}

# Get feedback data
cursor.execute("SELECT activity_title, rating FROM feedback")
feedback_rows = cursor.fetchall()

# Build training data
X = []
y = []
for activity_title, rating in feedback_rows:
    emb = title_to_embedding.get(activity_title)
    if emb is not None:
        X.append(emb)
        y.append(rating)

X = np.array(X)
y = np.array(y)

if len(X) > 0:
    # Train model
    model = RandomForestRegressor()
    model.fit(X, y)

    # Save model
    with open("ml_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Retrained model on {len(X)} feedback samples.")
else:
    print("Not enough feedback data to retrain the model.")

conn.close()