"""
Evaluates the trained model on the held-out test split and saves the
ACTUAL measured metrics + confusion matrix to results/.

Usage:
    python -m src.evaluate
"""
import json

import numpy as np
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from src.config import CACHE_DIR, MODEL_PATH, MODEL_CONFIG_PATH, RESULTS_DIR


def main():
    X_test = np.load(CACHE_DIR / "X_test.npy")
    y_test = np.load(CACHE_DIR / "y_test.npy")

    with open(MODEL_CONFIG_PATH) as f:
        cfg = json.load(f)
    threshold = cfg["decision_threshold"]

    model = keras.models.load_model(MODEL_PATH)
    probs_fake = model.predict(X_test, batch_size=64).ravel()
    preds = (probs_fake >= threshold).astype(int)

    metrics = {
        "threshold_used": threshold,
        "test_set_size": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs_fake)),
    }

    cm = confusion_matrix(y_test, preds)  # rows=actual [Real,Fake], cols=pred [Real,Fake]

    print(json.dumps(metrics, indent=2))
    print("\nConfusion matrix (rows=actual, cols=predicted; order=[Real, Fake]):")
    print(cm)
    print("\n", classification_report(y_test, preds, target_names=["Real", "Fake"]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump({
            **metrics,
            "confusion_matrix": {
                "actual_real_pred_real": int(cm[0, 0]),
                "actual_real_pred_fake": int(cm[0, 1]),
                "actual_fake_pred_real": int(cm[1, 0]),
                "actual_fake_pred_fake": int(cm[1, 1]),
            },
        }, f, indent=2)
    print(f"\nSaved metrics to {RESULTS_DIR / 'evaluation_metrics.json'}")
    print("These are the ACTUAL measured results from this run - not fabricated.")


if __name__ == "__main__":
    main()
