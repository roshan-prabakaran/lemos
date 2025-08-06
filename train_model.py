import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# -------------------------------
# Simulate Dataset
# -------------------------------
np.random.seed(42)
data = {
    "methane": np.random.randint(100, 800, 300),  # Methane levels in ppm
    "humidity": np.random.uniform(20, 90, 300)    # Humidity levels in %
}
df = pd.DataFrame(data)

# -------------------------------
# Define Danger Condition
# -------------------------------
# Danger = 1 if methane > 500 ppm and humidity > 60%
df["danger"] = ((df["methane"] > 250) & (df["humidity"] > 60)).astype(int)

# -------------------------------
# Prepare Features and Target
# -------------------------------
X = df[["methane", "humidity"]]
y = df["danger"]

# -------------------------------
# Train Random Forest Model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------------
# Save Model to File
# -------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")

