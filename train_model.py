import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Simulate dataset
np.random.seed(42)
data = {
    "methane": np.random.randint(100, 800, 300),  # Methane levels
    "humidity": np.random.uniform(20, 90, 300)    # Humidity levels
}
df = pd.DataFrame(data)

# Define 'danger' if methane > 500 ppm and humidity > 60%
df["danger"] = ((df["methane"] > 500) & (df["humidity"] > 60)).astype(int)

# Split features and target
X = df[["methane", "humidity"]]
y = df["danger"]

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model to file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
