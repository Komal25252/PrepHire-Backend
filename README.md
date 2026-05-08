# PrepHire ML Backend

Flask API serving resume classification, emotion analysis, and speech transcription for PrepHire.

## Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/predict` | Upload a PDF resume → returns domain + confidence |
| POST | `/analyze-emotion` | Send base64 frame → returns detected emotion |
| POST | `/transcribe` | Upload audio file → returns transcribed text |

## Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Server runs on `http://localhost:5000`

## Deployment (Render)

1. Push this folder as its own GitHub repo
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect the repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Python Version:** 3.11
5. Use at least the **Starter plan** (Whisper + DeepFace need >512MB RAM)

## Models

The `.pkl` model files are tracked via **Git LFS**. Make sure Git LFS is installed before cloning:

```bash
git lfs install
git clone <repo-url>
```
