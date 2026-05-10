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
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import gc

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ── Paths ────────────────────────────────────────────────────────────────────
_ML_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_ML_DIR, 'resume-class-ml')

# ── Resume classifier (Lazy Loading) ──────────────────────────────────────────
_classifier = None
_vectorizer = None
_encoder    = None

def load_resume_models():
    global _classifier, _vectorizer, _encoder
    if _classifier is None:
        print(">>> Lazy loading Resume Classifier (mmap_mode='r')...")
        _classifier = joblib.load(os.path.join(_MODEL_DIR, "resume_classifier_model.pkl"), mmap_mode='r')
        _vectorizer = joblib.load(os.path.join(_MODEL_DIR, "tfidf_vectorizer.pkl"), mmap_mode='r')
        _encoder    = joblib.load(os.path.join(_MODEL_DIR, "label_encoder.pkl"), mmap_mode='r')
        gc.collect()
    return _classifier, _vectorizer, _encoder

# ── HSEmotion ONNX model (Ultra Lightweight) ──────────────────────────────────
print(">>> Loading HSEmotion ONNX model...")
# Using a small efficientnet_b0 model for best speed/memory balance
fer_model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')

# AffectNet 8-class label mapping (used by enet_b0_8_best_afew)
# We map them to the 7 classes used by the frontend if needed
# Standard HSEmotion 8 labels: anger, contempt, disgust, fear, happy, neutral, sad, surprise
HSE_LABELS = fer_model.idx_to_class

# OpenCV face detector (ships with opencv-python-headless, no extra download)
_HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Whisper STT (Optimized to 'tiny') ─────────────────────────────────────
print(">>> Loading faster-whisper tiny model...")
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Explicitly clear memory after loading models
gc.collect()
print(">>> Model loading complete. GC collected.")


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
    Run HSEmotion ONNX on a single face crop.
    Returns (dominant_emotion, scores_dict)
    """
    # HSEmotion handle resizing and normalization internally
    emotion, scores = fer_model.predict_emotions(img_rgb, logits=False)
    
    # Standardize labels to lowercase for frontend consistency
    scores_dict = {}
    for label, score in zip(fer_model.idx_to_class.values(), scores):
        l = label.lower()
        if l == 'sad': l = 'sadness'
        scores_dict[l] = float(score * 100)
    
    dom_emotion = emotion.lower()
    if dom_emotion == 'sad': dom_emotion = 'sadness'
    
    return dom_emotion, scores_dict


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
        clf, vec_model, enc = load_resume_models()
        
        pdf_reader = PyPDF2.PdfReader(file)
        raw_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

        cleaned_text = clean_resume(raw_text)
        vec          = vec_model.transform([cleaned_text])
        pred         = clf.predict(vec)
        domain_name  = enc.inverse_transform(pred)[0]

        probs      = clf.predict_proba(vec)
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
    # Disable debug mode in production to save memory
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
