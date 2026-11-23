"""Paths and tunable settings. Everything lives on the local disk."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = Path(os.environ.get("FACEID_MODELS_DIR", ROOT / "models"))
DATA_DIR = Path(os.environ.get("FACEID_DATA_DIR", ROOT / "data"))
FRONTEND_DIST = ROOT / "frontend" / "dist"

DETECTOR_MODEL = MODELS_DIR / "det_10g.onnx"
EMBEDDER_MODEL = MODELS_DIR / "w600k_r50.onnx"

DB_PATH = DATA_DIR / "faceid.sqlite3"
INDEX_DIR = DATA_DIR / "indexes"
UPLOAD_DIR = DATA_DIR / "uploads"

EMBED_DIM = 512

# --- Detection -------------------------------------------------------------
DET_SCORE_THRESHOLD = 0.5
DET_NMS_THRESHOLD = 0.4
# Reference photos are usually one large face: a small detector input is enough.
ENROLL_DET_SIZE = 640
# Group photos have many small faces: run the detector at a higher resolution.
GROUP_DET_SIZE = 1280
# Faces smaller than this (pixels, shorter box side) are too small to trust.
MIN_FACE_SIZE = 20

# --- Matching --------------------------------------------------------------
# Cosine similarity between ArcFace embeddings. Calibrated on practice data
# (see RESEARCH.md): same-person pairs sit well above this, strangers below.
MATCH_THRESHOLD = 0.35
# Below the threshold but above this, we still say "unknown" and show the
# closest name as a hint so the user can judge.
HINT_THRESHOLD = 0.25

# --- Gateway / security ----------------------------------------------------
HOST = "127.0.0.1"  # loopback only: never reachable from another machine
PORT = int(os.environ.get("FACEID_PORT", "8000"))
# Set FACEID_DEV=1 only while developing the UI with the Vite dev server.
DEV_MODE = os.environ.get("FACEID_DEV") == "1"

MAX_REFERENCE_BYTES = 25 * 1024 * 1024   # one reference photo
MAX_GROUP_BYTES = 40 * 1024 * 1024       # one group photo
MAX_REQUEST_BYTES = MAX_GROUP_BYTES + 1024 * 1024
MAX_PEOPLE = 2000                        # people per folder
MAX_FILENAME_CHARS = 255
MAX_CONCURRENT_INFERENCE = 2             # keeps the laptop responsive

RATE_IDENTIFY_PER_MIN = 30
RATE_ENROL_PER_MIN = 3000
RATE_SESSION_PER_MIN = 60
RATE_GENERAL_PER_MIN = 3000

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".jfif",
}
