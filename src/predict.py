"""
Command-line single-image prediction. Useful for quick testing without
starting the web server.

Usage:
    python -m src.predict path/to/image.jpg
"""
import json
import sys

import numpy as np
from tensorflow import keras

from src.config import MODEL_PATH, MODEL_CONFIG_PATH, INV_LABEL_MAP
from src.preprocessing import load_and_preprocess_image, to_batch


def predict_image(path: str) -> dict:
    with open(MODEL_CONFIG_PATH) as f:
        cfg = json.load(f)
    threshold = cfg["decision_threshold"]
    img_size = cfg["img_size"]

    model = keras.models.load_model(MODEL_PATH)
    arr = load_and_preprocess_image(path, img_size=img_size)
    batch = to_batch(arr)

    prob_fake = float(model.predict(batch, verbose=0)[0, 0])
    label_idx = int(prob_fake >= threshold)
    label = INV_LABEL_MAP[label_idx]
    confidence = prob_fake * 100 if label == "FAKE" else (1 - prob_fake) * 100

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "probability_fake": round(prob_fake, 4),
        "threshold_used": threshold,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict path/to/image.jpg")
        sys.exit(1)
    result = predict_image(sys.argv[1])
    print(json.dumps(result, indent=2))
