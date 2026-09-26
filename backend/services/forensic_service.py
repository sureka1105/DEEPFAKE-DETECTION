"""
Computes real, pixel-level forensic signals from the actual uploaded image.

IMPORTANT HONESTY CONSTRAINT: every number here is calculated from the
actual image bytes the user uploaded. Nothing is hard-coded or templated
per label. These signals are WEAK, individually-inconclusive heuristics
(this is true of ELA/FFT/noise-residual analysis in general, not specific
to this project) - they are exposed as supporting evidence only, and are
combined conservatively. They must never be used to write a "definitely a
virtual camera" style claim; all wording stays hedged and proportional to
what the numbers actually show.
"""
import io
import numpy as np
from PIL import Image, ImageFilter


def _to_gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def _compute_ela(img: Image.Image, quality: int = 90) -> float:
    """Mean absolute pixel difference after a JPEG re-save, scaled 0-1."""
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    orig_arr = np.asarray(rgb, dtype=np.float64)
    resaved_arr = np.asarray(resaved, dtype=np.float64)
    diff = np.abs(orig_arr - resaved_arr)
    return float(diff.mean() / 255.0)


def _compute_fft_ratio(gray: np.ndarray) -> float:
    """Ratio of high-frequency to low-frequency energy in the 2D FFT spectrum."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2) + 1e-8

    low_mask = dist <= 0.15 * max_dist
    high_mask = dist >= 0.5 * max_dist

    low_energy = magnitude[low_mask].mean() if low_mask.any() else 1e-8
    high_energy = magnitude[high_mask].mean() if high_mask.any() else 0.0
    return float(high_energy / (low_energy + 1e-8))


def _compute_noise_residual_variance(gray: np.ndarray) -> float:
    """Variance of (image - gaussian_blur(image)); a proxy for sensor-noise floor."""
    gray_img = Image.fromarray(gray.astype(np.uint8))
    blurred = gray_img.filter(ImageFilter.GaussianBlur(radius=2))
    residual = gray - np.asarray(blurred, dtype=np.float64)
    return float(residual.var())


def _compute_edge_variance(gray: np.ndarray) -> float:
    """Laplacian-variance edge/sharpness metric (a common blur-detection signal)."""
    gray_img = Image.fromarray(gray.astype(np.uint8))
    laplacian_kernel = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)
    lap = gray_img.filter(laplacian_kernel)
    return float(np.asarray(lap, dtype=np.float64).var())


def analyze_forensics(image_bytes: bytes) -> dict:
    """Runs the full multi-signal pixel analysis on the raw uploaded bytes."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.load()
        gray = _to_gray_array(img)

        ela_error = _compute_ela(img)
        fft_ratio = _compute_fft_ratio(gray)
        noise_variance = _compute_noise_residual_variance(gray)
        edge_variance = _compute_edge_variance(gray)

    return {
        "ela_mean_error": round(ela_error, 4),
        "fft_high_low_ratio": round(fft_ratio, 4),
        "noise_residual_variance": round(noise_variance, 2),
        "edge_sharpness_variance": round(edge_variance, 2),
    }


def build_signal_notes(signals: dict) -> dict:
    """
    Turns raw numbers into short, hedged, human-readable notes. Thresholds
    below are rough rules of thumb from general image-forensics practice,
    NOT values fitted/validated on this project's own dataset - they are
    labeled as indicative only and are never the sole basis for the final
    verdict (the trained classifier's probability is).
    """
    ela = signals["ela_mean_error"]
    fft_ratio = signals["fft_high_low_ratio"]
    noise_var = signals["noise_residual_variance"]
    edge_var = signals["edge_sharpness_variance"]

    ela_note = (
        "Elevated re-compression error (may indicate synthetic/heavily edited "
        "content, or simply a low-quality/re-saved JPEG)."
        if ela > 0.08 else
        "Re-compression error within a typical range for an ordinary photo."
    )

    fft_note = (
        "Unusually high high-frequency content relative to low-frequency content, "
        "a pattern sometimes seen in GAN/diffusion outputs but also in sharpened "
        "or upscaled photos."
        if fft_ratio > 0.35 else
        "Frequency-spectrum balance consistent with typical natural images."
    )

    noise_note = (
        "Very low pixel-level noise floor - can indicate a smoothed/rendered "
        "image, but also a denoised or heavily compressed photo."
        if noise_var < 8 else
        "Noise floor consistent with an ordinary camera sensor or web-compressed photo."
    )

    edge_note = (
        "Sharper edge transitions than typical photos - seen in some AI-rendered "
        "images, but also in sharpened, illustrated, or vector-like content."
        if edge_var > 600 else
        "Edge sharpness within a typical range."
    )

    return {"ela_note": ela_note, "fft_note": fft_note, "noise_note": noise_note, "edge_note": edge_note}


def heuristic_probability_fake(signals: dict) -> float:
    """
    A transparent, evenly-weighted fallback score used ONLY when no trained
    model is available, so the app can still run end-to-end. This is
    intentionally conservative (centered near 0.5, nudged by evidence) so it
    does not systematically over-call FAKE or REAL the way a hard-coded
    template would.
    """
    ela = signals["ela_mean_error"]
    fft_ratio = signals["fft_high_low_ratio"]
    noise_var = signals["noise_residual_variance"]
    edge_var = signals["edge_sharpness_variance"]

    def norm(value, lo, hi):
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    score = (
        0.25 * norm(ela, 0.02, 0.15)
        + 0.25 * norm(fft_ratio, 0.10, 0.60)
        + 0.25 * (1 - norm(noise_var, 3, 25))   # low noise -> higher fake score
        + 0.25 * norm(edge_var, 200, 900)
    )
    return round(float(score), 4)
