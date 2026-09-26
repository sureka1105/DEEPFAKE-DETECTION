"""
Central configuration for the Deepfake Detection System.

Uses project-relative paths (pathlib) so the project works regardless
of where it is placed on disk or what the Windows username is.
"""
from pathlib import Path
import os

# BASE_DIR = the project root (parent of this "src" folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Dataset -----------------------------------------------------------
# Expected structure:
#   <BASE_DIR>/dataset/Final Dataset/Real/*.jpg
#   <BASE_DIR>/dataset/Final Dataset/Fake/*.jpg
#
# If you placed "Final Dataset" directly under the project root instead,
# both locations are checked automatically (see resolve_dataset_dir()).
DATASET_DIR_CANDIDATES = [
    BASE_DIR / "dataset" / "Final Dataset",
    BASE_DIR / "Final Dataset",
]

REAL_DIR_NAME = "Real"
FAKE_DIR_NAME = "Fake"

CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

MODEL_PATH = MODELS_DIR / "scratch_cnn_deepfake_detector.keras"
CHECKPOINT_PATH = MODELS_DIR / "checkpoint_scratch_cnn.keras"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

# ---- Model / training ----------------------------------------------------
IMG_SIZE = 128            # 128 x 128 input, as specified
CHANNELS = 3
BATCH_SIZE = 32
# Overridable via TRAIN_EPOCHS env var for time-boxed runs on limited
# hardware (e.g. a single-core sandbox). Default stays 25 for normal use.
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 25))
SEED = 42

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Label mapping (fixed, must match across train/eval/predict)
LABEL_MAP = {"Real": 0, "Fake": 1}
INV_LABEL_MAP = {0: "REAL", 1: "FAKE"}

# Decision threshold on P(fake). This MUST stay at 0.50 (the natural
# midpoint of a sigmoid binary classifier) unless you have validated a
# different cut-off against a held-out test set using evaluate.py.
#
# NOTE: a previous version of this file set this to 0.10, meaning any
# image the model was even 10% unsure about got labeled FAKE. That is the
# single biggest cause of real photos being wrongly flagged as fake -
# it has nothing to do with the image itself, it is a threshold bug.
DEFAULT_DECISION_THRESHOLD = 0.50


def resolve_dataset_dir() -> Path:
    """Find the actual dataset directory, trying known candidate locations."""
    for candidate in DATASET_DIR_CANDIDATES:
        if (candidate / REAL_DIR_NAME).exists() and (candidate / FAKE_DIR_NAME).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the dataset. Expected a 'Final Dataset' folder containing "
        "'Real' and 'Fake' subfolders under one of:\n"
        + "\n".join(str(c) for c in DATASET_DIR_CANDIDATES)
        + "\nPlace your dataset there, or update DATASET_DIR_CANDIDATES in src/config.py."
    )


for _d in (CACHE_DIR, MODELS_DIR, RESULTS_DIR, REPORTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
