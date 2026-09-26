"""
Trains the scratch CNN on the cached, preprocessed dataset produced by
build_cache.py.

Usage:
    python -m src.train

Requires: cache/X_train.npy, y_train.npy, X_val.npy, y_val.npy
(run build_cache.py first).
"""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras as keras_v3  # tf.keras is a compatibility shim without .saving; the standalone keras v3 package has it

from src.config import (
    CACHE_DIR, MODELS_DIR, MODEL_PATH, CHECKPOINT_PATH, MODEL_CONFIG_PATH,
    IMG_SIZE, BATCH_SIZE, EPOCHS, SEED, LABEL_MAP, DEFAULT_DECISION_THRESHOLD,
)
from src.model_def import build_scratch_cnn, build_transfer_model

tf.random.set_seed(SEED)


def load_cache(split):
    # mmap_mode='r' keeps the array file-backed (pages read on demand and
    # reclaimable by the OS under memory pressure) instead of copying the
    # whole array into the process's own heap - this sandbox only has ~4GB
    # RAM and passing full in-memory arrays straight into Keras .fit() was
    # getting OOM-killed (Keras/TF makes an internal copy on top of ours).
    X = np.load(CACHE_DIR / f"X_{split}.npy", mmap_mode="r")
    y = np.load(CACHE_DIR / f"y_{split}.npy")
    return X, y


def make_dataset(X, y, batch_size, shuffle, seed):
    n = len(y)

    def gen():
        order = np.arange(n)
        if shuffle:
            np.random.RandomState(seed).shuffle(order)
        for i in order:
            yield X[i], y[i]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(2000, n), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


@keras_v3.saving.register_keras_serializable(package="deepfake_detector")
class RandomJPEGRecompression(keras.layers.Layer):
    """
    Randomly re-encodes each training image through a low-to-mid quality
    JPEG pass. Messaging/social apps (WhatsApp, Instagram, etc.) do this to
    every photo people send - it wipes out a lot of the pixel-level noise
    and compression fingerprints the model may otherwise lean on. Without
    ever training on recompressed images, the model has no reason to be
    robust to it - this is the direct fix for that gap, not a threshold
    tweak.
    """
    def __init__(self, min_quality=35, max_quality=95, **kwargs):
        super().__init__(**kwargs)
        self.min_quality = min_quality
        self.max_quality = max_quality

    def call(self, images, training=None):
        if not training:
            return images

        def _recompress(img):
            img_uint8 = tf.image.convert_image_dtype(tf.clip_by_value(img, 0.0, 1.0), tf.uint8)
            img_jpeg = tf.image.random_jpeg_quality(img_uint8, self.min_quality, self.max_quality)
            return tf.image.convert_image_dtype(img_jpeg, tf.float32)

        return tf.map_fn(_recompress, images, fn_output_signature=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"min_quality": self.min_quality, "max_quality": self.max_quality})
        return config


def make_augmenter():
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
        RandomJPEGRecompression(min_quality=35, max_quality=95),
    ], name="augmentation")


def main():
    X_train, y_train = load_cache("train")
    X_val, y_val = load_cache("val")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True, seed=SEED).repeat()
    val_ds = make_dataset(X_val, y_val, BATCH_SIZE, shuffle=False, seed=SEED)
    steps_per_full_epoch = int(np.ceil(len(y_train) / BATCH_SIZE))
    # STEPS_PER_CALL lets one logical "epoch" be split across several
    # shorter, safer calls on a bigger dataset (defaults to a full epoch's
    # worth of steps, preserving old behavior when not set).
    steps_per_call = int(os.environ.get("STEPS_PER_CALL", steps_per_full_epoch))

    # Class weights to counter the Real/Fake imbalance (~5890 vs ~7000).
    n_real = int((y_train == 0).sum())
    n_fake = int((y_train == 1).sum())
    total = n_real + n_fake
    class_weight = {
        0: total / (2 * n_real),
        1: total / (2 * n_fake),
    }
    print("Class weights:", class_weight)

    augmenter = make_augmenter()

    STATE_PATH = MODELS_DIR / "train_state.json"
    state = {}
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
    prior_best_auc = state.get("best_val_auc")
    steps_completed = state.get("steps_completed", state.get("epochs_completed", 0) * steps_per_full_epoch)
    epochs_completed = steps_completed // steps_per_full_epoch  # for display/logging only

    # Cosine-decay schedule keyed off total STEPS actually trained so far
    # (persisted across every resumed run), not epochs - epochs can vary in
    # length now that STEPS_PER_CALL allows partial-epoch chunks, so steps
    # are the only consistent unit to schedule against.
    TOTAL_PLANNED_EPOCHS = 30
    TOTAL_PLANNED_STEPS = TOTAL_PLANNED_EPOCHS * steps_per_full_epoch
    PEAK_LR = 1e-3
    MIN_LR = 1e-6

    def cosine_lr(step_index: int) -> float:
        progress = min(step_index / TOTAL_PLANNED_STEPS, 1.0)
        return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + np.cos(np.pi * progress))

    current_lr = cosine_lr(steps_completed)

    # RESUME=1 continues from the last checkpoint (same architecture/weights)
    # instead of starting over - lets a long training run be split across
    # several shorter, safer sessions on limited hardware.
    if os.environ.get("RESUME") == "1" and CHECKPOINT_PATH.exists():
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH} (epochs completed so far: {epochs_completed})")
        loaded_train_model = keras.models.load_model(
            CHECKPOINT_PATH, custom_objects={"RandomJPEGRecompression": RandomJPEGRecompression}
        )
        # Rebuild the wrapper with the CURRENT augmenter code (not whatever
        # augmenter was saved inside the old checkpoint) so augmentation
        # changes (like adding JPEG-recompression robustness) actually take
        # effect on resume - only the inner CNN's learned weights carry over.
        model = loaded_train_model.layers[-1]

        if os.environ.get("UNFREEZE_BASE") == "1":
            # Second-phase fine-tuning: unfreeze every backbone layer up to
            # and including 'out_relu' (everything from MobileNetV2); layers
            # after that are the classification head we added and were
            # already trainable. Use a low LR_OVERRIDE with this (e.g. 1e-5)
            # - unfreezing at a high LR destroys the pretrained features.
            found_out_relu = False
            for layer in model.layers:
                layer.trainable = True
                if layer.name == "out_relu":
                    found_out_relu = True
                    break
            if found_out_relu:
                print("Unfroze backbone layers through 'out_relu' for fine-tuning.")
            else:
                print("UNFREEZE_BASE=1 set but this isn't a transfer model - ignoring.")

        inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = augmenter(inputs)
        outputs = model(x)
        train_model = keras.Model(inputs, outputs)
        train_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=current_lr),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )
        lr_override = os.environ.get("LR_OVERRIDE")
        if lr_override:
            train_model.optimizer.learning_rate.assign(float(lr_override))
            print(f"Learning rate manually overridden to {lr_override} for this run.")
        else:
            print(f"Cosine-schedule learning rate for epoch {epochs_completed}: {current_lr:.6f}")
    else:
        architecture = os.environ.get("MODEL_ARCHITECTURE", "scratch")
        if architecture == "transfer":
            model = build_transfer_model(trainable_base=os.environ.get("UNFREEZE_BASE") == "1")
            print("Building transfer-learning model (MobileNetV2 + ImageNet weights).")
        else:
            model = build_scratch_cnn()
        inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = augmenter(inputs)
        outputs = model(x)
        train_model = keras.Model(inputs, outputs)

        train_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=PEAK_LR),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )
        epochs_completed = 0
        steps_completed = 0

    class PersistBestAUC(keras.callbacks.Callback):
        """Updates the cross-run best-val_auc state file after every epoch
        (not just at the end of fit()) so a run that gets killed mid-way
        never leaves a stale/incorrect 'best so far' for the next resume."""
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_auc = logs.get("val_auc")
            if val_auc is None:
                return
            current_state = {}
            if STATE_PATH.exists():
                with open(STATE_PATH) as f:
                    current_state = json.load(f)
            current_best = current_state.get("best_val_auc", prior_best_auc or 0.0)
            current_steps = current_state.get("steps_completed", steps_completed)
            with open(STATE_PATH, "w") as f:
                json.dump({
                    "best_val_auc": max(current_best, val_auc),
                    "steps_completed": current_steps + steps_per_call,
                }, f)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(CHECKPOINT_PATH), monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
            initial_value_threshold=prior_best_auc,  # don't overwrite a better prior checkpoint
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6,
            restore_best_weights=True, verbose=1,
        ),
        PersistBestAUC(),
    ]

    history = train_model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_call,
        epochs=1,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Persist final state (PersistBestAUC already updates this per-epoch,
    # this is a final consistency pass in case fit() completed cleanly).
    best_this_run = max(history.history.get("val_auc", [0.0]))
    overall_best = max(best_this_run, prior_best_auc or 0.0)
    with open(STATE_PATH, "w") as f:
        json.dump({
            "best_val_auc": overall_best,
            "steps_completed": steps_completed + steps_per_call,
        }, f)

    # Load the best checkpoint across ALL runs (including prior resumed
    # runs) rather than trusting in-memory weights, which only reflect the
    # best epoch of *this* run.
    if CHECKPOINT_PATH.exists():
        best_train_model = keras.models.load_model(
            CHECKPOINT_PATH, custom_objects={"RandomJPEGRecompression": RandomJPEGRecompression}
        )
        model = best_train_model.layers[-1]

    # Save the underlying (non-augmented) inference model - augmentation
    # layers must not run at inference time.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}")

    # Detect architecture from the actual model layers (works whether this
    # run built fresh or resumed from a checkpoint) rather than assuming
    # scratch - a hardcoded label here would mislabel a transfer-learned
    # model as "trained from scratch" in the config the UI/PDF read from.
    is_transfer = any(layer.name == "out_relu" for layer in model.layers)
    if is_transfer:
        architecture_label = "transfer_mobilenetv2 (ImageNet-pretrained backbone, fine-tuned on this dataset)"
        last_conv_layer_name = "out_relu"
    else:
        architecture_label = "scratch_cnn (trained from scratch, not a pretrained backbone)"
        last_conv_layer_name = "last_conv"

    model_config = {
        "img_size": IMG_SIZE,
        "label_map": LABEL_MAP,
        "decision_threshold": DEFAULT_DECISION_THRESHOLD,
        "architecture": architecture_label,
        "last_conv_layer_name": last_conv_layer_name,
    }
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved model config to {MODEL_CONFIG_PATH}")


if __name__ == "__main__":
    main()
