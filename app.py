from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "model", "best_dt_model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "scaler.joblib")

model  = None
scaler = None

def load_model():
    global model, scaler

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        model = None

    # Load scaler (optional — only used if the file exists)
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("[INFO] Scaler loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Could not load scaler: {e}")
            scaler = None
    else:
        print("[WARN] No scaler file found at model/scaler.joblib — inputs will NOT be scaled.")

load_model()

# Exact feature order as in training data (is_safe is the target, excluded)
FEATURES = [
    "aluminium",
    "ammonia",
    "arsenic",
    "barium",
    "cadmium",
    "chloramine",
    "chromium",
    "copper",
    "flouride",
    "bacteria",
    "viruses",
    "lead",
    "nitrates",
    "nitrites",
    "mercury",
    "perchlorate",
    "radium",
    "selenium",
    "silver",
    "uranium",
]

# Metadata: label, unit, placeholder, danger threshold, safe_max for UI range hint
FEATURE_META = {
    "aluminium":   {"label": "Aluminium",   "unit": "mg/L",  "placeholder": "e.g. 1.4",    "danger": 2.8,  "icon": "⚗️"},
    "ammonia":     {"label": "Ammonia",     "unit": "mg/L",  "placeholder": "e.g. 16.0",   "danger": 32.5, "icon": "🧪"},
    "arsenic":     {"label": "Arsenic",     "unit": "mg/L",  "placeholder": "e.g. 0.005",  "danger": 0.01, "icon": "☠️"},
    "barium":      {"label": "Barium",      "unit": "mg/L",  "placeholder": "e.g. 1.0",    "danger": 2.0,  "icon": "🔬"},
    "cadmium":     {"label": "Cadmium",     "unit": "mg/L",  "placeholder": "e.g. 0.003",  "danger": 0.005,"icon": "⚠️"},
    "chloramine":  {"label": "Chloramine",  "unit": "mg/L",  "placeholder": "e.g. 2.0",    "danger": 4.0,  "icon": "🧴"},
    "chromium":    {"label": "Chromium",    "unit": "mg/L",  "placeholder": "e.g. 0.05",   "danger": 0.1,  "icon": "🌋"},
    "copper":      {"label": "Copper",      "unit": "mg/L",  "placeholder": "e.g. 0.6",    "danger": 1.3,  "icon": "🟤"},
    "flouride":    {"label": "Fluoride",    "unit": "mg/L",  "placeholder": "e.g. 0.8",    "danger": 1.5,  "icon": "💧"},
    "bacteria":    {"label": "Bacteria",    "unit": "count", "placeholder": "e.g. 0",      "danger": 0,    "icon": "🦠"},
    "viruses":     {"label": "Viruses",     "unit": "count", "placeholder": "e.g. 0",      "danger": 0,    "icon": "🔴"},
    "lead":        {"label": "Lead",        "unit": "mg/L",  "placeholder": "e.g. 0.008",  "danger": 0.015,"icon": "🩶"},
    "nitrates":    {"label": "Nitrates",    "unit": "mg/L",  "placeholder": "e.g. 5.0",    "danger": 10.0, "icon": "🌿"},
    "nitrites":    {"label": "Nitrites",    "unit": "mg/L",  "placeholder": "e.g. 0.5",    "danger": 1.0,  "icon": "🌱"},
    "mercury":     {"label": "Mercury",     "unit": "mg/L",  "placeholder": "e.g. 0.001",  "danger": 0.002,"icon": "🌡️"},
    "perchlorate": {"label": "Perchlorate", "unit": "mg/L",  "placeholder": "e.g. 28.0",   "danger": 56.0, "icon": "⚡"},
    "radium":      {"label": "Radium",      "unit": "pCi/L", "placeholder": "e.g. 2.5",    "danger": 5.0,  "icon": "☢️"},
    "selenium":    {"label": "Selenium",    "unit": "mg/L",  "placeholder": "e.g. 0.25",   "danger": 0.5,  "icon": "💊"},
    "silver":      {"label": "Silver",      "unit": "mg/L",  "placeholder": "e.g. 0.05",   "danger": 0.1,  "icon": "🪙"},
    "uranium":     {"label": "Uranium",     "unit": "mg/L",  "placeholder": "e.g. 0.15",   "danger": 0.3,  "icon": "☣️"},
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=FEATURES, meta=FEATURE_META)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Ensure best_dt_model.joblib is in the /model directory."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received."}), 400

        values = []
        for feat in FEATURES:
            raw = data.get(feat)
            if raw is None or str(raw).strip() == "":
                return jsonify({"error": f"Missing value for: {FEATURE_META[feat]['label']}"}), 400
            try:
                values.append(float(raw))
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid numeric value for: {FEATURE_META[feat]['label']}"}), 400

        input_array = np.array(values).reshape(1, -1)

        # Scale features if a scaler was loaded
        if scaler is not None:
            input_array = scaler.transform(input_array)
        else:
            print("[WARN] Predicting without scaling — results may be inaccurate.")

        prediction = int(model.predict(input_array)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_array)[0]
            confidence = round(float(max(proba)) * 100, 1)

        # Flag which parameters exceed danger thresholds
        flagged = []
        for i, feat in enumerate(FEATURES):
            threshold = FEATURE_META[feat]["danger"]
            if values[i] > threshold:
                flagged.append({
                    "label": FEATURE_META[feat]["label"],
                    "value": values[i],
                    "unit":  FEATURE_META[feat]["unit"],
                    "threshold": threshold,
                })

        inputs_summary = {
            FEATURE_META[f]["label"]: f"{values[i]} {FEATURE_META[f]['unit']}"
            for i, f in enumerate(FEATURES)
        }

        return jsonify({
            "prediction": prediction,
            "label": "SAFE" if prediction == 1 else "UNSAFE",
            "confidence": confidence,
            "flagged": flagged,
            "inputs": inputs_summary,
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)