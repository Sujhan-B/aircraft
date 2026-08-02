"""
Minimal backend that serves your trained crack/dent classifier to the
web app. The browser can't run the YOLO model directly, so this is the
bridge: index.html -> script.js -> POST /predict -> this server -> model.

Setup:
    pip install flask flask-cors ultralytics

Run:
    python app.py
Then open index.html in your browser (the "Scan for damage" button
calls this server at http://localhost:5000/predict).
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)  # allows index.html (opened as a local file) to call this server

MODEL_PATH = "runs_damage/crack_dent_cls/weights/best.pt"
model = YOLO(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    tmp_path = os.path.join("tmp_upload.jpg")
    file.save(tmp_path)

    results = model(tmp_path)
    r = results[0]
    probs = r.probs
    label = r.names[probs.top1]
    confidence = probs.top1conf.item()

    os.remove(tmp_path)

    return jsonify({"label": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
