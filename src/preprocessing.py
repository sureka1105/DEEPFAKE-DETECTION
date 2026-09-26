"""
Shared image preprocessing so training and inference always agree.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

from src.config import IMG_SIZE


def load_and_preprocess_image(path: Path | str, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Load an image from disk, fix orientation via EXIF, convert to RGB,
    resize to (img_size, img_size), and scale pixel values to [0, 1].

    Returns a float32 array of shape (img_size, img_size, 3).
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)  # respect EXIF orientation
        img = img.convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def preprocess_image_bytes(image_bytes: bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    """Same as load_and_preprocess_image but from in-memory bytes (used by the API)."""
    import io
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def to_batch(arr: np.ndarray) -> np.ndarray:
    """Add a batch dimension: (H, W, C) -> (1, H, W, C)."""
    return np.expand_dims(arr, axis=0)
