import sys
import json
import re
import joblib
import PyPDF2
import io
import os
import numpy as np

def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def main():
    # Read PDF bytes from stdin
    pdf_bytes = sys.stdin.buffer.read()
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    cleaned = clean_resume(raw_text)

    # Load models from ML directory (where this script lives)
    base = os.path.dirname(os.path.abspath(__file__))
    classifier = joblib.load(os.path.join(base, "resume_classifier_model.pkl"))
    vectorizer = joblib.load(os.path.join(base, "tfidf_vectorizer.pkl"))
    encoder    = joblib.load(os.path.join(base, "label_encoder.pkl"))

    vec = vectorizer.transform([cleaned])
    pred = classifier.predict(vec)
    domain = encoder.inverse_transform(pred)[0]

    # Also return confidence
    probs = classifier.predict_proba(vec)
    confidence = float(np.max(probs) * 100)

    # Return domain + first 3000 chars of raw text for context
    print(json.dumps({
        "domain": domain,
        "confidence": round(confidence, 1),
        "resumeText": raw_text[:3000].strip()
    }))

if __name__ == "__main__":
    main()
