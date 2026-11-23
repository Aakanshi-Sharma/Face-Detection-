"""One-time model download (the only step that needs internet).

Fetches the InsightFace `buffalo_l` model pack and keeps the two files the app
uses:

  det_10g.onnx     SCRFD-10GF face detector            (~17 MB)
  w600k_r50.onnx   ArcFace ResNet-50 face embedder     (~166 MB)

After this, the app runs fully offline. Uses only the Python standard library.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
WANTED = ("det_10g.onnx", "w600k_r50.onnx")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if all((MODELS_DIR / n).exists() for n in WANTED):
        print("Models already present in", MODELS_DIR)
        return 0

    zip_path = MODELS_DIR / "buffalo_l.zip.part"
    print("Downloading", URL)
    with urllib.request.urlopen(URL) as resp, open(zip_path, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last = -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            pct = int(done * 100 / total) if total else 0
            if pct != last and pct % 5 == 0:
                print(f"  {pct:3d}%  ({done >> 20} MB)", flush=True)
                last = pct

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in WANTED:
                with zf.open(member) as src, open(MODELS_DIR / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print("Extracted", name)
    zip_path.unlink()

    missing = [n for n in WANTED if not (MODELS_DIR / n).exists()]
    if missing:
        print("ERROR: missing after extraction:", missing)
        return 1
    print("Done. Models are in", MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
