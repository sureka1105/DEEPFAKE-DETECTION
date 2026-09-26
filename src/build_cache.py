"""
Builds a preprocessed, split, cached version of the dataset so train.py /
evaluate.py don't need to re-decode 12,890 images on every run.

IMPORTANT - GROUP-AWARE SPLITTING: this dataset contains augmented copies
of the same source photo (e.g. real_100_aug_0 .. real_100_aug_4 are all
derived from one original). A naive random split would put siblings of the
same photo in both train and test, letting the model "cheat" by recognizing
near-duplicates instead of learning a real signal - producing inflated
metrics that would NOT hold up on genuinely new images. This file groups
all augmented copies of the same base photo together before splitting, so
no source photo appears in more than one split.

Produces (under <BASE_DIR>/cache/):
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    split_manifest.json   (which original file went to which split - for
                            traceability; the original dataset is never
                            modified or moved)

Usage:
    python -m src.build_cache
"""
import json
import re

import numpy as np
from tqdm import tqdm

from src.config import (
    resolve_dataset_dir, REAL_DIR_NAME, FAKE_DIR_NAME, LABEL_MAP,
    CACHE_DIR, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SEED, IMG_SIZE,
)
from src.preprocessing import load_and_preprocess_image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
AUG_SUFFIX_RE = re.compile(r"^(.*)_aug_\d+$")


def group_key(path) -> str:
    """Files that are augmented copies of the same source photo share a
    group key (the filename with the '_aug_<N>' suffix stripped). Files
    with no such suffix are their own singleton group."""
    stem = path.stem
    m = AUG_SUFFIX_RE.match(stem)
    return m.group(1) if m else stem


def collect_class_files(dataset_dir, class_dir_name):
    folder = dataset_dir / class_dir_name
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in VALID_EXTS)


def split_class_by_group(files, seed):
    """Groups files by source photo, shuffles GROUPS (not files), then cuts
    into train/val/test so every file from one source photo lands in
    exactly one split."""
    groups: dict[str, list] = {}
    for f in files:
        groups.setdefault(group_key(f), []).append(f)

    group_keys = sorted(groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_train = int(round(n * TRAIN_SPLIT))
    n_val = int(round(n * VAL_SPLIT))

    train_keys = group_keys[:n_train]
    val_keys = group_keys[n_train:n_train + n_val]
    test_keys = group_keys[n_train + n_val:]

    def flatten(keys):
        out = []
        for k in keys:
            out.extend(groups[k])
        return out

    return flatten(train_keys), flatten(val_keys), flatten(test_keys)


def build_array_shuffled(items):
    """items: list of (path, label) tuples, already in final shuffled order.
    Writes directly into one pre-sized array (no intermediate per-class
    arrays, no concatenate/fancy-index copy) to keep peak memory low."""
    X = np.zeros((len(items), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    y = np.zeros((len(items),), dtype=np.int64)
    kept_paths = []
    write_i = 0
    for path, label in tqdm(items, desc="Preprocessing"):
        try:
            X[write_i] = load_and_preprocess_image(path)
            y[write_i] = label
            kept_paths.append(str(path))
            write_i += 1
        except Exception as e:
            print(f"Skipping unreadable file {path}: {e}")
    return X[:write_i], y[:write_i], kept_paths


def main():
    dataset_dir = resolve_dataset_dir()
    real_files = collect_class_files(dataset_dir, REAL_DIR_NAME)
    fake_files = collect_class_files(dataset_dir, FAKE_DIR_NAME)
    print(f"Found {len(real_files)} Real, {len(fake_files)} Fake images")

    real_train, real_val, real_test = split_class_by_group(real_files, SEED)
    fake_train, fake_val, fake_test = split_class_by_group(fake_files, SEED)
    print(f"Real groups -> train {len(real_train)}, val {len(real_val)}, test {len(real_test)} files")
    print(f"Fake groups -> train {len(fake_train)}, val {len(fake_val)}, test {len(fake_test)} files")

    manifest = {}
    for split_name, real_paths, fake_paths in (
        ("train", real_train, fake_train),
        ("val", real_val, fake_val),
        ("test", real_test, fake_test),
    ):
        items = [(p, LABEL_MAP["Real"]) for p in real_paths] + \
                [(p, LABEL_MAP["Fake"]) for p in fake_paths]
        rng = np.random.RandomState(SEED)
        rng.shuffle(items)  # shuffle the (cheap) list of paths, not arrays

        X, y, kept_paths = build_array_shuffled(items)
        np.save(CACHE_DIR / f"X_{split_name}.npy", X)
        np.save(CACHE_DIR / f"y_{split_name}.npy", y)
        manifest[split_name] = kept_paths
        print(f"Saved cache/X_{split_name}.npy {X.shape}, cache/y_{split_name}.npy {y.shape}")
        del X, y

    with open(CACHE_DIR / "split_manifest.json", "w") as f:
        json.dump(manifest, f)

    print("\nCache build complete (group-aware split - no source photo appears in more than one split).")
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
