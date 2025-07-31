from flask import Flask, request, jsonify, render_template
import csv
import datetime
import os

app = Flask(__name__)

DATA_FILE = "data/emissions.csv"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)


@app.route("/api/sensor-data", methods=["POST"])
def receive_data():
    data = request.get_json()
    data["timestamp"] = datetime.datetime.now().isoformat()

    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "methane", "humidity"])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(data)

    print("Received:", data)
    return jsonify({"status": "received"}), 200


@app.route("/api/data")
def get_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return f.read()
    return "No data", 200


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
