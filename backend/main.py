"""
Deepfake Detection System - FastAPI backend.

Run (from the `backend/` folder, with your Python 3.11 venv active):

    python -m uvicorn main:app --reload

Endpoints:
    GET  /api/health
    POST /api/analyze         (multipart/form-data, field name "file")
    POST /api/report/{report_id}   (returns the PDF for a previous analysis)
"""
import sys
import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.services.inference_service import get_inference_service  # noqa: E402
from backend.services.gradcam_service import GradCamService  # noqa: E402
from backend.services.metadata_service import extract_metadata  # noqa: E402
from backend.services.explanation_service import (  # noqa: E402
    build_authenticity_assessment, build_explanation, DISCLAIMER,
)
from backend.services.report_service import build_pdf_report  # noqa: E402
from backend.services.forensic_service import (  # noqa: E402
    analyze_forensics, build_signal_notes, heuristic_probability_fake,
)
from backend.services.camera_profile_service import build_camera_profile  # noqa: E402
from backend.schemas.response_models import HealthResponse  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepfake-api")

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# ---- CORS: allow the local Vite dev server ------------------------------
FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# In-memory store of the most recent analyses, keyed by report_id, so the
# "Download PDF" button can fetch the report without re-uploading the image.
# (No database is used, per project scope - this is intentionally ephemeral.)
_recent_analyses: dict[str, dict] = {}
_MAX_CACHED_REPORTS = 50

_inference_service = None
_gradcam_service = None


@app.on_event("startup")
def load_model_on_startup():
    """Load the trained model once when the server starts (never retrains)."""
    global _inference_service, _gradcam_service
    try:
        _inference_service = get_inference_service()
        _gradcam_service = GradCamService(_inference_service.model)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model failed to load: {e}")
        _inference_service = None
        _gradcam_service = None


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=bool(_inference_service and _inference_service.is_loaded()),
    )


def _validate_upload(file: UploadFile, content: bytes):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG, JPEG, PNG, or WEBP image.",
        )
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unable to analyze this image. Please upload a valid JPG, JPEG, PNG, or WEBP image.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="File is too large. Maximum allowed size is 10 MB.",
        )
    # Verify it's actually a decodable image, not just a renamed file.
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="This file could not be read as a valid image. It may be corrupted or an unsupported format.",
        )


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    _validate_upload(file, content)

    model_available = _inference_service is not None and _inference_service.is_loaded()

    # 1. Real, pixel-level forensic signals - computed from THIS image only,
    #    never templated by label. Never allowed to crash the whole request.
    try:
        forensic_signals = analyze_forensics(content)
        forensic_notes = build_signal_notes(forensic_signals)
    except Exception:
        logger.exception("Forensic signal analysis failed")
        forensic_signals, forensic_notes = {}, {}

    # 2. Prediction. Prefer the trained CNN; only fall back to the
    #    transparent heuristic fusion above if no trained model is loaded
    #    yet (so the app stays usable before training completes), and this
    #    is clearly labeled as such in the response - never silently swapped in.
    heuristic_score = heuristic_probability_fake(forensic_signals) if forensic_signals else None
    gradcam_result = {"original": "", "heatmap": "", "overlay": "", "available": False}
    if model_available:
        try:
            pred_result = _inference_service.predict_from_bytes(content)
            prediction = pred_result["prediction"]
            confidence = pred_result["confidence"]
            probability_fake = pred_result["probability_fake"]
            model_source = "trained_cnn"
            try:
                gradcam_result = _gradcam_service.generate(pred_result["preprocessed_array"])
            except Exception:
                logger.exception("Grad-CAM failed")
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Unable to analyze this image. Please try again.")
    else:
        if heuristic_score is None:
            raise HTTPException(
                status_code=503,
                detail="The AI model is not currently available and the fallback "
                       "analysis could not be run. Please try again shortly.",
            )
        probability_fake = heuristic_score
        prediction = "FAKE" if probability_fake >= 0.5 else "REAL"
        confidence = round((probability_fake if prediction == "FAKE" else 1 - probability_fake) * 100, 2)
        model_source = "heuristic_fallback"

    # 3. Metadata (never allowed to crash the whole request)
    try:
        metadata = extract_metadata(content, file.filename or "uploaded_image")
    except Exception:
        logger.exception("Metadata extraction failed")
        metadata = {
            "filename": file.filename or "uploaded_image",
            "file_size_bytes": len(content),
            "observations": ["Metadata could not be extracted. This does not affect the AI image classification."],
        }

    # 4. Authenticity assessment + explanation (grounded in the above only)
    assessment = build_authenticity_assessment(prediction)
    explanation = build_explanation(
        prediction, confidence, metadata, gradcam_result.get("available", True),
        model_source=model_source,
    )
    camera_profile = build_camera_profile(metadata, forensic_signals, heuristic_score)

    report_id = uuid.uuid4().hex

    response = {
        "prediction": prediction,
        "confidence": confidence,
        "probability_fake": probability_fake,
        "authenticity_assessment": assessment,
        "explanation": explanation,
        "gradcam": gradcam_result,
        "metadata": metadata,
        "forensics": {**forensic_signals, **forensic_notes},
        "camera_profile": camera_profile,
        "model_source": model_source,
        "disclaimer": DISCLAIMER,
        "report_id": report_id,
    }

    # Cache for PDF generation, with a simple size cap.
    if len(_recent_analyses) >= _MAX_CACHED_REPORTS:
        _recent_analyses.pop(next(iter(_recent_analyses)))
    _recent_analyses[report_id] = response

    return response


@app.post("/api/report/{report_id}")
def download_report(report_id: str):
    analysis = _recent_analyses.get(report_id)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="This report has expired. Please re-analyze the image to download a fresh report.",
        )
    try:
        pdf_bytes = build_pdf_report(analysis)
    except Exception:
        logger.exception("PDF generation failed")
        raise HTTPException(status_code=500, detail="Unable to generate the PDF report. Please try again.")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="authenticity_report_{report_id[:8]}.pdf"'},
    )
