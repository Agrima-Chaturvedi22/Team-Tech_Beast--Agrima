"""
KindlyHeart backend
--------------------
A small Flask server that:

1. Trains a real scikit-learn Logistic Regression model on heart.csv
   at startup (the same dataset used in heart_disease_prediction.ipynb),
   using the 9 fields the frontend form actually collects.
2. Exposes a REST API:
      GET  /api/health              -> server + model status
      POST /api/predict             -> {probability, risk} for given patient data
      POST /api/patients            -> save a patient's details
      GET  /api/patients            -> list saved patients (for demo purposes)
3. Serves the existing frontend (index.html, patient-details.html, etc.)
   from the same server, so the whole project runs from one command.

Run it with:
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000/patient-details.html
"""

import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..")  # index.html etc. live one level up
CSV_PATH = os.path.join(BASE_DIR, "heart.csv")
DB_PATH = os.path.join(BASE_DIR, "kindlyheart.db")

# The 9 fields the website's form actually collects (see index.html).
# The full UCI dataset also has oldpeak/slope/ca/thal, but the frontend
# doesn't ask for those, so the backend model is trained on this subset
# to match exactly what the site can send it.
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang"]

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------------------
# Model training (runs once, at server startup)
# ---------------------------------------------------------------------------
def train_model():
    df = pd.read_csv(CSV_PATH)

    X = df[FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"[KindlyHeart] Model trained on {len(df)} rows.")
    print(f"[KindlyHeart] Training accuracy: {train_acc:.3f}")
    print(f"[KindlyHeart] Test accuracy:     {test_acc:.3f}")

    return model, train_acc, test_acc


MODEL, TRAIN_ACC, TEST_ACC = train_model()


# ---------------------------------------------------------------------------
# Database (stores patients submitted from patient-details.html)
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            email TEXT,
            address TEXT,
            city TEXT,
            pincode TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL is not None,
            "train_accuracy": round(TRAIN_ACC, 3),
            "test_accuracy": round(TEST_ACC, 3),
            "rows_trained_on": int(pd.read_csv(CSV_PATH).shape[0]),
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        row = np.array([[float(payload[f]) for f in FEATURES]])
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric."}), 400

    prediction = int(MODEL.predict(row)[0])
    probability = float(MODEL.predict_proba(row)[0][1])  # P(target == 1)

    return jsonify(
        {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk": "high" if prediction == 1 else "low",
        }
    )


@app.route("/api/patients", methods=["POST"])
def add_patient():
    data = request.get_json(silent=True) or {}
    required = ["fullName", "age", "gender", "phone", "address", "city", "pincode"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO patients (full_name, age, gender, phone, email, address, city, pincode, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("fullName"),
            data.get("age"),
            data.get("gender"),
            data.get("phone"),
            data.get("email", ""),
            data.get("address"),
            data.get("city"),
            data.get("pincode"),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    patient_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    return jsonify({"status": "saved", "id": patient_id}), 201


@app.route("/api/patients", methods=["GET"])
def list_patients():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Serve the existing frontend from the same server
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "patient-details.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
