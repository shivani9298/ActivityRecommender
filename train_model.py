import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

X_train = np.random.rand(200, 384)
y_train = np.random.rand(200)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
with open("ml_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("ML model trained and saved to ml_model.pkl")
