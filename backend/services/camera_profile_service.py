"""
Builds the "Camera & Device Detection" table shown in the UI/PDF.

HONESTY CONSTRAINT: every field is derived from something actually
observed (real EXIF data) or explicitly marked as an unconfirmed heuristic
inference tied to the actual signal that produced it. Nothing here is a
fixed string repeated for every FAKE (or every REAL) result - a previous
version of this project did that (e.g. always writing "Virtual / AI
Synthesis Engine" regardless of the image) and that is exactly the kind of
false-confidence text this file must never produce again.
"""

# Common exact output resolutions for popular diffusion/generator tools
# (Stable Diffusion, SDXL, Midjourney, DALL-E, etc). A camera photo landing
# on one of these exact dimensions is possible but comparatively unusual -
# camera sensors produce fixed native resolutions with their own aspect
# ratios (4:3, 3:2, 16:9), and messaging apps that resize photos also tend
# to use their own resize targets, not these specific values. This is a
# real, checkable pattern - not a certainty.
_COMMON_GENERATOR_RESOLUTIONS = {
    (512, 512), (512, 768), (768, 512), (576, 1024), (1024, 576),
    (640, 640), (768, 768), (896, 896), (960, 960), (1024, 1024),
    (1024, 1536), (1536, 1024), (1152, 896), (896, 1152), (1344, 768),
    (768, 1344), (1216, 832), (832, 1216),
}


def _resolution_note(width, height):
    if not width or not height:
        return None
    if (width, height) in _COMMON_GENERATOR_RESOLUTIONS:
        return (
            f"{width}x{height} is a common exact output size for AI image "
            f"generators (e.g. Stable Diffusion/SDXL/Midjourney presets). "
            f"Camera photos and typical resizes rarely land on this exact "
            f"value - this is a supporting clue, not proof on its own."
        )
    return None


def build_camera_profile(metadata: dict, forensic_signals: dict, heuristic_score: float | None) -> dict:
    has_exif_camera = bool(metadata.get("camera_make") or metadata.get("camera_model"))
    width, height = metadata.get("width"), metadata.get("height")
    resolution_note = _resolution_note(width, height)

    if has_exif_camera:
        capture_source = "Camera-Captured Photo (EXIF Present)"
        sensor_profile = "EXIF indicates a physical camera; noise signal below is supporting evidence only."
    elif resolution_note:
        # Always surface this real, checkable fact when no EXIF is present -
        # its wording reflects whether the pixel-noise heuristic agrees or
        # disagrees, rather than hiding the resolution fact either way.
        if heuristic_score is not None and heuristic_score <= 0.35:
            capture_source = "Mixed signals: generator-typical resolution, but pixel noise reads camera-like"
        else:
            capture_source = "Possibly Synthetic / AI-Generated (no EXIF + generator-typical resolution)"
        sensor_profile = resolution_note
    elif heuristic_score is not None and heuristic_score >= 0.65:
        capture_source = "Possibly Synthetic / AI-Generated (unconfirmed heuristic signal)"
        sensor_profile = "Low noise / high-frequency pattern noted below is a weak indicator, not proof."
    elif heuristic_score is not None and heuristic_score <= 0.35:
        capture_source = "No EXIF found, but pixel signals resemble a typical camera photo"
        sensor_profile = "Noise and edge signals fall within a typical camera-photo range."
    else:
        capture_source = "Undetermined (no EXIF, pixel signals inconclusive)"
        sensor_profile = "Signals were not distinctive enough to draw a conclusion."

    size_kb = round((metadata.get("file_size_bytes") or 0) / 1024, 1)
    channels = metadata.get("channels")
    color_mode = metadata.get("color_mode")
    gamut = f"{color_mode} ({channels * 8}-bit, {channels} channel{'s' if channels != 1 else ''})" if channels else (color_mode or "Unknown")

    profile = {
        "capture_source": capture_source,
        "sensor_profile": sensor_profile,
        "dimensions_size": f"{width} x {height} px ({size_kb} KB)" if width else "Unknown",
        "color_gamut_depth": gamut,
        "gps_location": "Present (not shown, for privacy)" if metadata.get("gps_available") else "None found",
    }

    if has_exif_camera:
        # Only show individual camera fields when there's actually EXIF
        # camera data to show - each of these is a real observed value.
        profile["camera_details"] = {
            "camera_make": metadata.get("camera_make") or "Present but unnamed",
            "camera_model": metadata.get("camera_model") or "Present but unnamed",
            "device_category": metadata.get("device_category") or "Camera (make/model present, category unclassified)",
            "lens_model": metadata.get("lens_model") or "Not recorded in EXIF",
            "focal_length_aperture": " · ".join(
                v for v in (metadata.get("focal_length"), metadata.get("aperture")) if v
            ) or "Not recorded in EXIF",
            "exposure_iso": " · ".join(
                v for v in (metadata.get("exposure_time"), metadata.get("iso")) if v
            ) or "Not recorded in EXIF",
            "capture_software": metadata.get("software") or "Not recorded in EXIF",
        }
        profile["no_exif_note"] = None
    else:
        # No repeated "Not available" per field - one honest, consolidated
        # line instead, since the underlying fact (no EXIF camera block
        # exists in this file) is a single fact, not seven separate ones.
        profile["camera_details"] = None
        profile["no_exif_note"] = (
            "No camera EXIF block (make, model, lens, exposure, capture "
            "software) was found in this file. This is expected for "
            "training-dataset images, screenshots, and most images that "
            "have passed through messaging or social apps, which strip "
            "this data - it is not itself a sign of anything."
        )

    return profile
