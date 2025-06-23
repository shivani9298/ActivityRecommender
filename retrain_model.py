import psycopg2
import os
import pickle
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor

# Load environment variables
load_dotenv()


with open("activity_embeddings.pkl", "rb") as f:
    activity_rows, activity_embeddings = pickle.load(f)

title_to_embedding = {row[1]: emb for row, emb in zip(activity_rows, activity_embeddings)}
print("Loaded activity embeddings.")

conn = psycopg2.connect(
    dbname=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD'),
    host=os.getenv('POSTGRES_HOST', 'localhost'),
    port=os.getenv('POSTGRES_PORT', '5432')
)
cursor = conn.cursor()

# Get all feedback entries 
cursor.execute("SELECT activity_title, rating FROM feedback")
feedback_rows = cursor.fetchall()
print(f"Found {len(feedback_rows)} feedback entries in the database.")


#Prepare the training data
X_train = []
y_train = []
matched_count = 0
unmatched_count = 0

for activity_title, rating in feedback_rows:
    # Clean the activity title from feedback

    
    if activity_title == "college_plan":
        unmatched_count += 1
        continue
    
    clean_title = activity_title
    if clean_title.startswith("- "):
        clean_title = clean_title[2:] 
    
    if " (score:" in clean_title:
        clean_title = clean_title.split(" (score:")[0]
    
    # Find the embedding
    embedding = title_to_embedding.get(clean_title)
    if embedding is not None and rating is not None:
        X_train.append(embedding)
        y_train.append(rating)
        matched_count += 1
        print(f"Matched: '{clean_title}' with rating {rating}")
    else:
        unmatched_count += 1
        print(f"Could not find embedding for: '{clean_title}' (from '{activity_title}') with rating {rating}")

print(f"Matched {matched_count} feedback entries to embeddings.")
print(f"Unmatched {unmatched_count} feedback entries.")

# Convert to NumPy arrays for scikit-learn
X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Final training data: {len(X_train)} samples")

# Only retrain if we have feedback data.
if len(X_train) > 0:
    print(f"Retraining model on {len(X_train)} feedback samples...")
    
    # Initialize and train the RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Overwrite the old model file with our new model
    with open("ml_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("Successfully retrained and saved the new model to ml_model.pkl.")
else:
    print("No new feedback data to retrain the model. The model was not updated.")

conn.close() 