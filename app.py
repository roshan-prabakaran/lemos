from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import datetime
import csv
import os
from twilio.rest import Client
from typing import Dict, Any, Optional

app = Flask(__name__, static_folder='static')

# --- Configuration ---
class Config:
    DATA_FILE = "data/emissions.csv"
    MODEL_FILE = "model.pkl"
    DATA_DIR = "data"
    STATIC_DIR = "static"
    MAX_READINGS = 1000  # Maximum readings to keep in memory

# --- Initialize directories ---
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.STATIC_DIR, exist_ok=True)

# --- Load ML model ---
try:
    with open(Config.MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print("✅ ML model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load ML model: {e}")
    model = None  # Handle case where model fails to load

# --- Twilio Alert System ---
class AlertSystem:
    def __init__(self):
        self.sid = os.environ.get("TWILIO_SID")
        self.auth = os.environ.get("TWILIO_AUTH")
        self.from_num = os.environ.get("TWILIO_FROM")
        self.to_num = os.environ.get("ALERT_TO")
        self.last_alert_time = None
        self.alert_cooldown = 3600  # 1 hour cooldown between alerts

    def send_alert(self, message: str) -> bool:
        """Send SMS alert with cooldown period"""
        if not all([self.sid, self.auth, self.from_num, self.to_num]):
            print("❌ Twilio credentials not configured")
            return False

        current_time = datetime.datetime.now()
        if (self.last_alert_time and 
            (current_time - self.last_alert_time).seconds < self.alert_cooldown):
            print("⏳ Alert cooldown active")
            return False

        try:
            client = Client(self.sid, self.auth)
            client.messages.create(
                body=message,
                from_=self.from_num,
                to=self.to_num
            )
            self.last_alert_time = current_time
            print("🚨 Alert sent successfully")
            return True
        except Exception as e:
            print(f"❌ Error sending SMS: {e}")
            return False

alert_system = AlertSystem()

# --- Data Management ---
class DataHandler:
    def __init__(self):
        self.cached_data = []
        self._load_initial_data()

    def _load_initial_data(self):
        """Load existing data from file at startup"""
        if os.path.exists(Config.DATA_FILE):
            with open(Config.DATA_FILE, "r") as f:
                reader = csv.DictReader(f)
                self.cached_data = list(reader)[-Config.MAX_READINGS:]
            print(f"📊 Loaded {len(self.cached_data)} existing readings")

    def add_reading(self, data: Dict[str, Any]) -> None:
        """Add new reading to cache and file"""
        self.cached_data.append(data)
        if len(self.cached_data) > Config.MAX_READINGS:
            self.cached_data = self.cached_data[-Config.MAX_READINGS:]

        # Write to file
        file_exists = os.path.exists(Config.DATA_FILE)
        with open(Config.DATA_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "methane", "humidity"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

    def get_all_readings(self) -> str:
        """Return all readings as CSV string"""
        if not self.cached_data:
            return "No data yet"

        output = ["timestamp,methane,humidity"]  # Header
        for reading in self.cached_data:
            output.append(f"{reading['timestamp']},{reading['methane']},{reading['humidity']}")
        return "\n".join(output)

data_handler = DataHandler()

# --- Routes ---
@app.route("/")
def home() -> str:
    """Render the dashboard page"""
    return render_template("index.html")

@app.route("/api/sensor-data", methods=["POST"])
def receive_data() -> tuple:
    """
    Receive sensor data, run prediction, and store it
    Returns:
        JSON response with status and danger prediction
    """
    data: Optional[Dict] = request.get_json()
    if not data or "methane" not in data or "humidity" not in data:
        return jsonify({"error": "Invalid data format"}), 400

    # Add timestamp and store data
    data["timestamp"] = datetime.datetime.now().isoformat()
    data_handler.add_reading(data)

    # Run ML prediction if model is available
    danger = 0
    if model:
        try:
            X_input = [[float(data["methane"]), float(data["humidity"])]]
            danger = int(model.predict(X_input)[0])
            
            # Send alert if danger detected
            if danger == 1:
                alert_system.send_alert(
                    f"⚠️ High landfill emission detected!\n"
                    f"Methane: {data['methane']} ppm\n"
                    f"Humidity: {data['humidity']}%\n"
                    f"Time: {data['timestamp']}"
                )
        except Exception as e:
            print(f"❌ Prediction error: {e}")

    return jsonify({
        "status": "received",
        "danger": danger,
        "timestamp": data["timestamp"]
    }), 200

@app.route("/api/data", methods=["GET"])
def get_data() -> tuple:
    """Return all sensor data as CSV"""
    return data_handler.get_all_readings(), 200

@app.route('/static/<path:path>')
def serve_static(path: str):
    """Serve static files"""
    return send_from_directory('static', path)

# --- Error Handlers ---
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
