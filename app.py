from flask import Flask, request, jsonify
import joblib
import re
import io
import PyPDF2 
from flask_cors import CORS 
import base64
from PIL import Image
import numpy as np
from deepface import DeepFace
import os
import whisper
import tempfile

# Ensure common paths for ffmpeg (especially for Mac Homebrew)
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin" + os.pathsep + "/usr/local/bin"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# 1. Load Models
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_ML_DIR, 'resume-class-ml')

classifier = joblib.load(os.path.join(_MODEL_DIR, "resume_classifier_model.pkl"))
vectorizer = joblib.load(os.path.join(_MODEL_DIR, "tfidf_vectorizer.pkl"))
encoder = joblib.load(os.path.join(_MODEL_DIR, "label_encoder.pkl"))

# Load Whisper model (base version is good for Mac)
print(">>> Loading Whisper base model...")
stt_model = whisper.load_model("base")

def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text) 
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        raw_text = ""
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
        
        cleaned_text = clean_resume(raw_text)
        vec = vectorizer.transform([cleaned_text])
        pred = classifier.predict(vec)
        domain_name = encoder.inverse_transform(pred)[0]

        probs = classifier.predict_proba(vec)
        confidence = float(np.max(probs) * 100)
        
        return jsonify({
            "domain": domain_name,
            "confidence": round(confidence, 1),
            "resumeText": raw_text[:3000].strip(),
            "status": "success"
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
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        results = DeepFace.analyze(np.array(img), actions=['emotion'], enforce_detection=False)
        
        if not results:
            return jsonify({"error": "No results from analysis"}), 500
            
        face_data = results[0]
        raw_scores = face_data['emotion']
        
        mapping = {
            'angry': 'anger',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sadness',
            'surprise': 'surprise'
        }
        
        final_scores = {mapping[k]: float(v) for k, v in raw_scores.items() if k in mapping}
        dominant = face_data['dominant_emotion']
        predicted_emotion = mapping.get(dominant, dominant)
        
        return jsonify({
            "emotion": predicted_emotion,
            "scores": final_scores
        })
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
        result = stt_model.transcribe(tmp_path)
        text = result["text"].strip()
        print(f">>> Whisper Output: {text}")
        return jsonify({
            "text": text,
            "status": "success"
        })
    except Exception as e:
        print(f">>> Transcription Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    app.run(port=5000, debug=True)