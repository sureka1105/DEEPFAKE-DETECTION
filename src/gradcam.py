"""
Grad-CAM: produces a heatmap of image regions that contributed most
strongly to the model's prediction.

This is an attention-visualization technique - NOT proof of manipulation.
"""
import json

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import MODEL_CONFIG_PATH


def _get_last_conv_layer_name(model):
    try:
        with open(MODEL_CONFIG_PATH) as f:
            cfg = json.load(f)
        name = cfg.get("last_conv_layer_name")
        if name and any(layer.name == name for layer in model.layers):
            return name
    except FileNotFoundError:
        pass
    # Fallback: last layer whose output has rank 4 (a feature map).
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not locate a convolutional layer for Grad-CAM.")


def compute_gradcam(model, image_batch: np.ndarray, layer_name: str | None = None) -> np.ndarray:
    """
    image_batch: shape (1, H, W, 3), already preprocessed to [0,1].
    Returns a (H, W) heatmap normalized to [0, 1].
    """
    if layer_name is None:
        layer_name = _get_last_conv_layer_name(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch)
        loss = predictions[:, 0]  # sigmoid output = P(fake)

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def make_overlay(original_rgb01: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    original_rgb01: (H, W, 3) float array in [0,1].
    heatmap: (h, w) float array in [0,1], smaller spatial size than original.
    Returns a PIL Image of the overlay at the original resolution.
    """
    h, w = original_rgb01.shape[:2]
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
    heatmap_arr = np.asarray(heatmap_img, dtype=np.float32) / 255.0

    # Simple red-hot colormap without external deps (matplotlib not required).
    colored = np.zeros((h, w, 3), dtype=np.float32)
    colored[..., 0] = heatmap_arr                      # red channel rises with intensity
    colored[..., 1] = np.clip(heatmap_arr - 0.5, 0, 1) * 2  # green kicks in at high intensity
    colored[..., 2] = np.clip(0.5 - heatmap_arr, 0, 1) * 2  # blue fades out

    original = original_rgb01.astype(np.float32)
    overlay = (1 - alpha) * original + alpha * colored
    overlay = np.clip(overlay, 0, 1)
    return Image.fromarray(np.uint8(overlay * 255))


def make_heatmap_image(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Standalone heatmap image (not overlaid), resized to `size` = (w, h)."""
    colored = np.zeros((*heatmap.shape, 3), dtype=np.float32)
    colored[..., 0] = heatmap
    colored[..., 1] = np.clip(heatmap - 0.5, 0, 1) * 2
    colored[..., 2] = np.clip(0.5 - heatmap, 0, 1) * 2
    img = Image.fromarray(np.uint8(colored * 255))
    return img.resize(size, Image.BILINEAR)
