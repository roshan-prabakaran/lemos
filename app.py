from flask import Flask, request, jsonify, render_template
import pickle
import datetime
import csv
import os
from twilio.rest import Client

app = Flask(__name__)

# --- File path ---
DATA_FILE = "data/emissions.csv"

# --- Create data folder if it doesn't exist ---
os.makedirs("data", exist_ok=True)

# --- Load ML model at startup ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Twilio credentials (NOTE: Move to .env or Render secrets in production) ---
import os

TWILIO_SID = os.environ.get("TWILIO_SID")
TWILIO_AUTH = os.environ.get("TWILIO_AUTH")
TWILIO_FROM = os.environ.get("TWILIO_FROM")
ALERT_TO = os.environ.get("ALERT_TO")

def send_alert(message):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=ALERT_TO
        )
        print("🚨 Alert sent successfully.")
    except Exception as e:
        print("❌ Error sending SMS:", e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/sensor-data", methods=["POST"])
def receive_data():
    data = request.get_json()
    data["timestamp"] = datetime.datetime.now().isoformat()

    # Run ML prediction
    X_input = [[data["methane"], data["humidity"]]]
    prediction = model.predict(X_input)[0]

    # Save to CSV
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "methane", "humidity"])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(data)

    # Trigger SMS if danger predicted
    if prediction == 1:
        send_alert("⚠️ High landfill emission detected! Check LEMOS dashboard.")

    return jsonify({
        "status": "received",
        "danger": int(prediction)
    }), 200

@app.route("/api/data", methods=["GET"])
def get_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return f.read()
    return "No data yet", 200

if __name__ == "__main__":
    app.run(debug=True)
