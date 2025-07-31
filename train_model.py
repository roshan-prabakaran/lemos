import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Simulate dataset
np.random.seed(42)
data = {
    "methane": np.random.randint(100, 800, 300),
    "humidity": np.random.uniform(20, 90, 300)
}
df = pd.DataFrame(data)

# Define 'toxic' if methane > 500 and humidity > 60
df["danger"] = ((df["methane"] > 500) & (df["humidity"] > 60)).astype(int)

# Train model
X = df[["methane", "humidity"]]
y = df["danger"]
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
