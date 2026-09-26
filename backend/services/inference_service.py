"""
Loads the trained model ONCE (singleton) and runs predictions.
No hard-coded predictions or confidence - every value comes from the
actual model output.
"""
import json
import sys
from pathlib import Path

import numpy as np

# Make the project root importable so we can reuse src/ code without duplication.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from tensorflow import keras  # noqa: E402
from src.config import MODEL_PATH, MODEL_CONFIG_PATH, INV_LABEL_MAP  # noqa: E402
from src.preprocessing import preprocess_image_bytes, to_batch  # noqa: E402


class InferenceService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                f"Run `python -m src.train` first."
            )
        with open(MODEL_CONFIG_PATH) as f:
            self.config = json.load(f)

        self.model = keras.models.load_model(MODEL_PATH)
        self.img_size = self.config["img_size"]
        self.threshold = self.config["decision_threshold"]
        self.loaded = True

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        arr = preprocess_image_bytes(image_bytes, img_size=self.img_size)
        batch = to_batch(arr)

        prob_fake = float(self.model.predict(batch, verbose=0)[0, 0])
        label_idx = int(prob_fake >= self.threshold)
        label = INV_LABEL_MAP[label_idx]
        confidence = prob_fake * 100 if label == "FAKE" else (1 - prob_fake) * 100

        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "probability_fake": round(prob_fake, 4),
            "preprocessed_array": arr,  # returned for reuse by Grad-CAM (avoids re-decoding)
        }

    def is_loaded(self) -> bool:
        return getattr(self, "loaded", False)


# Convenience singleton accessor used by main.py
def get_inference_service() -> InferenceService:
    return InferenceService()
