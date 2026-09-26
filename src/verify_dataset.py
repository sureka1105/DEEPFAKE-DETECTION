"""
Scans the dataset and reports basic health info: counts, unreadable files,
and duplicate filenames across classes. Run this before build_cache.py.

Usage:
    python -m src.verify_dataset
"""
from PIL import Image

from src.config import resolve_dataset_dir, REAL_DIR_NAME, FAKE_DIR_NAME

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def scan_class(folder):
    files = [f for f in folder.iterdir() if f.suffix.lower() in VALID_EXTS]
    bad = []
    for f in files:
        try:
            with Image.open(f) as img:
                img.verify()
        except Exception:
            bad.append(f)
    return files, bad


def main():
    dataset_dir = resolve_dataset_dir()
    print(f"Dataset directory: {dataset_dir}")

    real_dir = dataset_dir / REAL_DIR_NAME
    fake_dir = dataset_dir / FAKE_DIR_NAME

    real_files, real_bad = scan_class(real_dir)
    fake_files, fake_bad = scan_class(fake_dir)

    print(f"Real images found: {len(real_files)}  (unreadable: {len(real_bad)})")
    print(f"Fake images found: {len(fake_files)}  (unreadable: {len(fake_bad)})")
    print(f"Total: {len(real_files) + len(fake_files)}")

    if real_bad:
        print("\nUnreadable Real files:")
        for f in real_bad[:20]:
            print(" -", f.name)
    if fake_bad:
        print("\nUnreadable Fake files:")
        for f in fake_bad[:20]:
            print(" -", f.name)

    if not real_bad and not fake_bad:
        print("\nAll files opened successfully.")


if __name__ == "__main__":
    main()
