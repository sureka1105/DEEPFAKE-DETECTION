"""
Defines the CNN architecture.

Two options:
- build_scratch_cnn(): a small CNN trained entirely from scratch on this
  project's own dataset. Fast, no internet required, but - because it has
  only ever seen ~9k images of one fairly narrow style - it generalizes
  poorly to images stylistically different from the training set ("random"
  photos from the wild).
- build_transfer_model(): a small classification head on top of a
  MobileNetV2 backbone PRETRAINED ON IMAGENET (1.4M diverse real-world
  photos). The backbone already knows general visual features (edges,
  textures, objects, lighting) before ever seeing a single deepfake image,
  which is what actually improves generalization to images unlike the
  training set - fine-tuning on top of that is a very different (and much
  stronger) starting point than learning everything from zero.
  Requires downloading ImageNet weights the first time (needs normal
  internet access - this only works in an environment that isn't network-
  restricted, e.g. your own machine, not this sandbox).
"""
from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMG_SIZE, CHANNELS


def build_scratch_cnn() -> keras.Model:
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Named so gradcam.py can find the last conv layer reliably.
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probability_fake")(x)

    model = keras.Model(inputs, outputs, name="scratch_cnn_deepfake_detector")
    return model


def build_transfer_model(trainable_base: bool = False) -> keras.Model:
    """
    MobileNetV2 backbone pretrained on ImageNet + a small classification
    head. Needs internet access to download ImageNet weights the first
    time it's built (not available in a network-restricted sandbox - run
    this on a machine with normal internet access).

    trainable_base=False (default): backbone frozen, only the new head
    trains - fast, good first step, low risk of destroying the pretrained
    features on a small dataset.
    trainable_base=True: unfreezes the backbone for fine-tuning - use this
    for a second training pass, at a low learning rate (e.g. 1e-5), AFTER
    the head has already been trained with the backbone frozen.

    Built via input_tensor (not called as a nested sub-layer) so
    MobileNetV2's own internal layers - including 'out_relu', the last
    spatial conv output Grad-CAM needs - stay reachable via
    model.get_layer(...) on the OUTER model, instead of being hidden
    inside one opaque nested layer.
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
    # MobileNetV2 expects inputs preprocessed to [-1, 1]; the rest of this
    # project's pipeline normalizes to [0, 1], so convert first.
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)

    base = keras.applications.MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights="imagenet",
        pooling=None,  # keep the spatial (4x4x1280) output for Grad-CAM
    )
    base.trainable = trainable_base

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probability_fake")(x)

    model = keras.Model(inputs, outputs, name="transfer_mobilenetv2_deepfake_detector")
    return model
