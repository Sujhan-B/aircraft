import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
from PIL import Image

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'webapp', 'runs_damage', 'crack_dent_cls', 'weights', 'best.onnx')
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
CLASS_NAMES = {0: 'crack', 1: 'dent'}


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    input_tensor = preprocess(file.read())

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    top_idx = int(np.argmax(probs))
    label = CLASS_NAMES.get(top_idx, str(top_idx))
    confidence = float(probs[top_idx])

    return jsonify({'label': label, 'confidence': confidence})


@app.route('/')
def index():
    return jsonify({'status': 'AeroScan API is running'})
