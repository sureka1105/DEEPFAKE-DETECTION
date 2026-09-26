"""
Builds the "AI Explanation" and "Authenticity Assessment" strings strictly
from the actual prediction/confidence/metadata already computed - never
invents reasons or artifacts.
"""

DISCLAIMER = (
    "This system provides a probabilistic AI-based authenticity assessment. "
    "AI confidence, metadata information, and Grad-CAM visualizations are "
    "supporting analysis and should not be treated as definitive proof that "
    "an image is authentic or manipulated."
)


def build_authenticity_assessment(prediction: str) -> str:
    return "Likely Manipulated" if prediction == "FAKE" else "Likely Authentic"


def build_explanation(
    prediction: str, confidence: float, metadata: dict, gradcam_available: bool,
    model_source: str = "trained_cnn",
) -> str:
    if model_source == "heuristic_fallback":
        parts = [
            f"No trained model is currently loaded, so this result comes from a "
            f"transparent fallback that combines several pixel-level forensic "
            f"signals (see the breakdown below). It classified this image as "
            f"{prediction} with an estimated confidence of {confidence}%. "
            f"This fallback is less reliable than a properly trained classifier "
            f"and should be treated with extra caution."
        ]
    else:
        parts = [
            f"The trained image classification model classified this image as "
            f"{prediction} with a confidence of {confidence}%."
        ]

    if gradcam_available:
        parts.append(
            "The Grad-CAM visualization highlights image regions that "
            "contributed strongly to the model's prediction."
        )
    elif model_source != "heuristic_fallback":
        parts.append("A visual (Grad-CAM) explanation was not available for this image.")

    observations = metadata.get("observations") or []
    if metadata.get("software"):
        parts.append(
            "Metadata observation: editing software information was detected in the "
            "file. This is supporting evidence only and does not independently prove manipulation."
        )
    elif not metadata.get("exif_available"):
        parts.append(
            "No EXIF metadata was found in this file. Missing metadata is common and "
            "does not, by itself, indicate manipulation."
        )

    parts.append(
        "This assessment is probabilistic, based on learned visual patterns, "
        "and is not definitive proof of authenticity or manipulation."
    )
    return " ".join(parts)
