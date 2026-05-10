from flask import Flask, request, jsonify
import joblib
import re
import io
import os
import base64
import tempfile

import cv2
import numpy as np
from PIL import Image
import PyPDF2
from flask_cors import CORS
from faster_whisper import WhisperModel
import tensorflow as tf

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ── Paths ────────────────────────────────────────────────────────────────────
_ML_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_ML_DIR, 'resume-class-ml')

# ── Resume classifier ────────────────────────────────────────────────────────
classifier = joblib.load(os.path.join(_MODEL_DIR, "resume_classifier_model.pkl"))
vectorizer = joblib.load(os.path.join(_MODEL_DIR, "tfidf_vectorizer.pkl"))
encoder    = joblib.load(os.path.join(_MODEL_DIR, "label_encoder.pkl"))

# ── MobileNet FER model ──────────────────────────────────────────────────────
print(">>> Loading MobileNet FER model...")
fer_model = tf.keras.models.load_model(
    os.path.join(_ML_DIR, "mobilenet_7.h5"), compile=False
)

# AffectNet 7-class label order used by mobilenet_7.h5
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

# OpenCV face detector (ships with opencv-python-headless, no extra download)
_HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Whisper STT ──────────────────────────────────────────────────────────────
print(">>> Loading faster-whisper base model...")
stt_model = WhisperModel("base", device="cpu", compute_type="int8")


# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_resume(text: str) -> str:
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def predict_emotion(img_rgb: np.ndarray):
    """
    Run MobileNet FER on a single face crop.
    img_rgb: H×W×3 uint8 numpy array (any size — will be resized to 224×224)
    Returns (dominant_emotion, scores_dict)
    """
    face_resized = cv2.resize(img_rgb, (224, 224))
    face_input   = np.expand_dims(face_resized.astype("float32") / 255.0, axis=0)
    preds        = fer_model.predict(face_input, verbose=0)[0]          # shape (7,)
    scores       = {EMOTION_LABELS[i]: float(preds[i] * 100) for i in range(7)}
    dominant     = EMOTION_LABELS[int(np.argmax(preds))]
    return dominant, scores


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        raw_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

        cleaned_text = clean_resume(raw_text)
        vec          = vectorizer.transform([cleaned_text])
        pred         = classifier.predict(vec)
        domain_name  = encoder.inverse_transform(pred)[0]

        probs      = classifier.predict_proba(vec)
        confidence = float(np.max(probs) * 100)

        return jsonify({
            "domain":     domain_name,
            "confidence": round(confidence, 1),
            "resumeText": raw_text[:3000].strip(),
            "status":     "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-emotion", methods=["POST"])
def analyze_emotion():
    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({"error": "frame field is missing"}), 400

    try:
        frame_str = data['frame']
        if "," in frame_str:
            frame_str = frame_str.split(",")[1]

        img_bytes = base64.b64decode(frame_str)
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_rgb   = np.array(img_pil)
        img_gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Detect faces with Haar cascade
        faces = _HAAR.detectMultiScale(
            img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        if len(faces) == 0:
            # No face detected — run on full frame (enforce_detection=False equivalent)
            dominant, scores = predict_emotion(img_rgb)
        else:
            # Use the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop   = img_rgb[y:y+h, x:x+w]
            dominant, scores = predict_emotion(face_crop)

        return jsonify({"emotion": dominant, "scores": scores})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        segments, _ = stt_model.transcribe(tmp_path, beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
        print(f">>> Whisper Output: {text}")
        return jsonify({"text": text, "status": "success"})
    except Exception as e:
        print(f">>> Transcription Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
