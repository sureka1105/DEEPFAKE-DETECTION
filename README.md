# Deepfake Detection System

An end-to-end image authenticity analyzer: upload a face image and get a
REAL/FAKE prediction, confidence score, Grad-CAM visual explanation,
metadata analysis, a plain-language AI explanation, and a downloadable
PDF report.

## Important — read before running

This project was written and syntax-checked in a sandbox **without**
TensorFlow, FastAPI, or internet access, so none of it has been executed
end-to-end yet. Every file compiles, but you must run it on your own
Windows machine (Python 3.11, with internet for `pip`/`npm install`) and
work through Phase 14 (testing) yourself. Do not treat any accuracy
numbers as real until you've run `src/evaluate.py` and seen the actual
output.

## 1. Architecture

```
USER → UPLOAD → VALIDATION → CNN MODEL → REAL/FAKE + CONFIDENCE
     → AUTHENTICITY ASSESSMENT → GRAD-CAM → METADATA ANALYSIS
     → AI EXPLANATION → RESULT DASHBOARD → PDF REPORT
```

- **ML pipeline** (`src/`): preprocessing, dataset caching, CNN training
  (from scratch — not a pretrained backbone), evaluation, Grad-CAM,
  CLI prediction.
- **Backend** (`backend/`): FastAPI app. Loads the trained model once at
  startup, exposes `/api/health` and `/api/analyze`, generates Grad-CAM
  and PDF reports on demand.
- **Frontend** (`frontend/`): React + Vite, plain CSS (no Tailwind, to
  keep dependencies minimal). Upload → preview → analyze → result
  dashboard → PDF download. No "About this project" pages — it's built
  as a real product, not a report.

## 2. Dataset

Your existing dataset (verified: 5,890 Real + 7,000 Fake = 12,890 images)
must be placed at one of these locations relative to the project root:

```
deepfake_detection/
├── dataset/
│   └── Final Dataset/
│       ├── Real/
│       └── Fake/
```
or directly as `deepfake_detection/Final Dataset/...`.

`src/config.py` checks both locations automatically — no path editing
needed, and the original dataset is never modified, moved, or deleted by
any script here.

## 3. Setup (Windows PowerShell)

Use your Python 3.11 environment (not Python 3.14, if you have both
installed):

```powershell
cd deepfake_detection
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Build the dataset cache, then train and evaluate

```powershell
python -m src.verify_dataset
python -m src.build_cache
python -m src.train
python -m src.evaluate
```

`train.py` saves:
- `models/scratch_cnn_deepfake_detector.keras` (final model)
- `models/checkpoint_scratch_cnn.keras` (best checkpoint during training)
- `models/model_config.json` (image size, label map, decision threshold,
  last-conv-layer name for Grad-CAM)

`evaluate.py` writes the **actual, measured** metrics and confusion
matrix to `results/evaluation_metrics.json` — nothing here is
pre-filled or assumed.

Quick single-image test without the web app:
```powershell
python -m src.predict "path\to\image.jpg"
```

## 4. Run the backend

```powershell
cd backend
python -m uvicorn main:app --reload
```
- Loads the trained model once at startup (never retrains).
- `GET  http://127.0.0.1:8000/api/health`
- `POST http://127.0.0.1:8000/api/analyze` (multipart field `file`)
- `POST http://127.0.0.1:8000/api/report/{report_id}` → PDF

## 5. Run the frontend

```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`. The Vite dev server proxies `/api` calls
to the backend at `127.0.0.1:8000` (see `vite.config.js`) — no CORS
configuration needed beyond what's already in `backend/main.py`.

## 6. Model

A small CNN trained from scratch (see `src/model_def.py`) — **not**
Xception, ResNet, or EfficientNet. Input size 128×128. Label mapping:
`Real = 0`, `Fake = 1`. The decision threshold (default `0.10`,
favoring recall on fake detection) is stored in `models/model_config.json`
and applied consistently by both `src/predict.py` and the backend.

## 7. Grad-CAM

`src/gradcam.py` computes a Grad-CAM heatmap from the last convolutional
layer and produces a heatmap image and an overlay. It is an attention
visualization, not proof of manipulation — the UI and PDF report both
say this explicitly.

## 8. Metadata analysis

`backend/services/metadata_service.py` extracts real EXIF/file data
(format, dimensions, camera make/model, editing software, GPS
*presence* only — never exact coordinates). Metadata is supporting
evidence only; it never decides REAL/FAKE by itself, and missing
metadata is never treated as evidence of fakery.

## 9. PDF report

`backend/services/report_service.py` (ReportLab) builds a PDF containing
the prediction, confidence, assessment, AI explanation, the three
Grad-CAM images, metadata table, and disclaimer — using the exact
values from that analysis, nothing fabricated.

## 10. Known limitations

- Not yet run end-to-end (see banner above) — you need to run Phase 14
  testing yourself and record real results.
- No database — PDF reports are generated on demand from an in-memory
  cache of recent analyses (last 50), keyed by `report_id`. Re-running
  the server clears this cache.
- Face detection/cropping is not implemented — the model expects the
  uploaded image to already be a face-centric image, matching the
  training data.

## 11. Disclaimer

This system provides a probabilistic AI-based authenticity assessment.
AI confidence, metadata information, and Grad-CAM visualizations are
supporting analysis and should not be treated as definitive proof that
an image is authentic or manipulated.
