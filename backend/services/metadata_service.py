"""
Extracts structured metadata from an uploaded image. Metadata is
SUPPORTING EVIDENCE ONLY - it never decides REAL/FAKE by itself.
"""
import io
from PIL import Image, ExifTags

# Reverse-lookup table so we can find tags by name instead of numeric id.
_TAG_NAME_TO_ID = {v: k for k, v in ExifTags.TAGS.items()}

GPS_TAG_ID = _TAG_NAME_TO_ID.get("GPSInfo")
SOFTWARE_TAG_ID = _TAG_NAME_TO_ID.get("Software")
MAKE_TAG_ID = _TAG_NAME_TO_ID.get("Make")
MODEL_TAG_ID = _TAG_NAME_TO_ID.get("Model")
DATETIME_TAG_ID = _TAG_NAME_TO_ID.get("DateTimeOriginal")
ORIENTATION_TAG_ID = _TAG_NAME_TO_ID.get("Orientation")
EXIF_IFD_POINTER = 0x8769  # points to the sub-IFD holding lens/exposure tags

# Tags that live in the Exif sub-IFD (not the base IFD above).
FOCAL_LENGTH_ID = 0x920A
FNUMBER_ID = 0x829D
EXPOSURE_TIME_ID = 0x829A
ISO_ID = 0x8827
LENS_MODEL_ID = 0xA434


def _phone_or_camera(make: str | None, model: str | None) -> str | None:
    text = f"{make or ''} {model or ''}".lower()
    if not text.strip():
        return None
    phone_brands = ["iphone", "samsung", "pixel", "xiaomi", "huawei", "oneplus", "oppo", "vivo", "redmi"]
    if any(b in text for b in phone_brands):
        return "Smartphone"
    return "Digital Camera"


def extract_metadata(image_bytes: bytes, filename: str) -> dict:
    file_size = len(image_bytes)
    observations: list[str] = []

    result = {
        "filename": filename,
        "format": None,
        "mime_type": None,
        "file_size_bytes": file_size,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "color_mode": None,
        "channels": None,
        "exif_available": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime_original": None,
        "orientation": None,
        "gps_available": False,
        "device_category": None,
        "lens_model": None,
        "focal_length": None,
        "aperture": None,
        "exposure_time": None,
        "iso": None,
        "observations": observations,
    }

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            result["format"] = img.format
            result["mime_type"] = Image.MIME.get(img.format) if img.format else None
            result["width"], result["height"] = img.size
            if result["height"]:
                result["aspect_ratio"] = round(result["width"] / result["height"], 4)
            result["color_mode"] = img.mode
            result["channels"] = len(img.getbands())

            exif = img.getexif()
            if exif and len(exif) > 0:
                result["exif_available"] = True

                if MAKE_TAG_ID in exif:
                    result["camera_make"] = str(exif[MAKE_TAG_ID]).strip() or None
                if MODEL_TAG_ID in exif:
                    result["camera_model"] = str(exif[MODEL_TAG_ID]).strip() or None
                if SOFTWARE_TAG_ID in exif:
                    sw = str(exif[SOFTWARE_TAG_ID]).strip()
                    result["software"] = sw or None
                    if sw:
                        observations.append("Editing software metadata was detected.")
                if DATETIME_TAG_ID in exif:
                    result["datetime_original"] = str(exif[DATETIME_TAG_ID])
                if ORIENTATION_TAG_ID in exif:
                    result["orientation"] = int(exif[ORIENTATION_TAG_ID])
                if GPS_TAG_ID in exif:
                    result["gps_available"] = True
                    observations.append("GPS metadata: Present.")  # never expose exact coords

                result["device_category"] = _phone_or_camera(result["camera_make"], result["camera_model"])

                # Lens/exposure tags live in the Exif sub-IFD, not the base IFD.
                try:
                    exif_ifd = exif.get_ifd(EXIF_IFD_POINTER)
                except Exception:
                    exif_ifd = {}
                if exif_ifd:
                    if LENS_MODEL_ID in exif_ifd:
                        result["lens_model"] = str(exif_ifd[LENS_MODEL_ID]).strip() or None
                    if FOCAL_LENGTH_ID in exif_ifd:
                        try:
                            result["focal_length"] = f"{float(exif_ifd[FOCAL_LENGTH_ID]):.1f}mm"
                        except (TypeError, ValueError):
                            pass
                    if FNUMBER_ID in exif_ifd:
                        try:
                            result["aperture"] = f"f/{float(exif_ifd[FNUMBER_ID]):.1f}"
                        except (TypeError, ValueError):
                            pass
                    if EXPOSURE_TIME_ID in exif_ifd:
                        try:
                            result["exposure_time"] = f"1/{round(1 / float(exif_ifd[EXPOSURE_TIME_ID]))}s"
                        except (TypeError, ValueError, ZeroDivisionError):
                            pass
                    if ISO_ID in exif_ifd:
                        try:
                            result["iso"] = f"ISO {int(exif_ifd[ISO_ID])}"
                        except (TypeError, ValueError):
                            pass
            else:
                observations.append(
                    "No EXIF metadata found. This is common for web-shared or "
                    "screenshotted images and does not, by itself, indicate manipulation."
                )
    except Exception as e:
        observations.append(f"Metadata could not be fully extracted: {e}")

    return result
