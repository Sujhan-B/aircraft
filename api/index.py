from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'webapp', 'runs_damage', 'crack_dent_cls', 'weights', 'best.pt')
model = YOLO(MODEL_PATH)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    tmp_path = '/tmp/tmp_upload.jpg'
    file.save(tmp_path)

    results = model(tmp_path)
    r = results[0]
    probs = r.probs
    label = r.names[probs.top1]
    confidence = probs.top1conf.item()

    os.remove(tmp_path)

    return jsonify({'label': label, 'confidence': confidence})


@app.route('/')
def index():
    return jsonify({'status': 'AeroScan API is running'})
