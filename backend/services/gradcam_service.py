"""
Generates Grad-CAM visual explanations and encodes them as base64 data URIs
so the frontend can render them directly with no extra file serving.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.gradcam import compute_gradcam, make_overlay, make_heatmap_image  # noqa: E402


def _pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


class GradCamService:
    def __init__(self, model):
        self.model = model

    def generate(self, preprocessed_array: np.ndarray) -> dict:
        """
        preprocessed_array: (H, W, 3) float32 in [0,1], the SAME array used
        for prediction (so the visualization matches what the model saw).
        """
        batch = np.expand_dims(preprocessed_array, axis=0)

        try:
            heatmap = compute_gradcam(self.model, batch)
            h, w = preprocessed_array.shape[:2]
            heatmap_img = make_heatmap_image(heatmap, size=(w, h))
            overlay_img = make_overlay(preprocessed_array, heatmap)
            original_img = Image.fromarray(np.uint8(preprocessed_array * 255))

            return {
                "original": _pil_to_data_uri(original_img),
                "heatmap": _pil_to_data_uri(heatmap_img),
                "overlay": _pil_to_data_uri(overlay_img),
                "available": True,
            }
        except Exception as e:
            # Grad-CAM failing must never crash the whole analysis.
            original_img = Image.fromarray(np.uint8(preprocessed_array * 255))
            placeholder = _pil_to_data_uri(original_img)
            return {
                "original": placeholder,
                "heatmap": placeholder,
                "overlay": placeholder,
                "available": False,
                "error": str(e),
            }
