"""Draw the 'how it works' figures used in README.md from real pipeline output.

  uv run --project backend python scripts/make_readme_figures.py

Needs practice_data/ (see make_practice_data.py). Writes to docs/images/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import faiss  # noqa: E402

from app import config, matching  # noqa: E402
from app.embedder import align_face  # noqa: E402
from app.engine import FaceEngine  # noqa: E402
from app.imaging import crop_face, decode_image, name_from_filename  # noqa: E402

DATA = ROOT / "practice_data"
OUT = ROOT / "docs" / "images"
PHOTO = DATA / "group_02.jpg"

KNOWN, UNKNOWN, INK, MUTED, LINE = (15, 138, 108), (185, 106, 6), (29, 28, 26), (106, 103, 95), (227, 224, 216)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = FaceEngine()
    engine.load()

    # Enrol the practice folder (in memory only).
    names, thumbs, vecs = {}, {}, []
    for i, path in enumerate(sorted((DATA / "people").iterdir())):
        bgr = decode_image(path.read_bytes())
        res = engine.analyse_reference(bgr)
        if res["status"] == "failed":
            continue
        names[i] = name_from_filename(path.name)
        thumbs[i] = cv2.imdecode(np.frombuffer(res["thumb"], np.uint8), 1)
        vecs.append((i, res["embedding"]))
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(config.EMBED_DIM))
    index.add_with_ids(np.stack([v for _, v in vecs]), np.array([i for i, _ in vecs], dtype=np.int64))

    bgr = decode_image(PHOTO.read_bytes())
    faces, emb = engine.analyse_group(bgr)
    person_ids, sims = matching.similarity_matrix(index, emb)
    results = matching.assign(person_ids, sims, names)

    # --- Step 1: detection (boxes + 5 landmarks) ---------------------------
    k = 2
    big = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    for f in faces:
        x1, y1, x2, y2 = (f.box * k).astype(int)
        cv2.rectangle(big, (x1, y1), (x2, y2), (255, 200, 60), 2, cv2.LINE_AA)
        for px, py in (f.landmarks * k).astype(int):
            cv2.circle(big, (px, py), 4, (60, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(big, (px, py), 4, (0, 0, 0), 1, cv2.LINE_AA)
    to_pil(big).save(OUT / "process-1-detect.jpg", quality=90)

    # --- Step 2: alignment (every face warped to the same 112x112 pose) -----
    tile, gap = 112, 10
    strip = Image.new("RGB", (len(faces) * (tile + gap) + gap, tile + 2 * gap + 22), "white")
    d = ImageDraw.Draw(strip)
    for i, f in enumerate(faces):
        x = gap + i * (tile + gap)
        strip.paste(to_pil(align_face(bgr, f.landmarks)), (x, gap))
        d.text((x + tile / 2, tile + gap + 12), f"face {i + 1}", font=font(13), fill=MUTED, anchor="mm")
    strip.save(OUT / "process-2-align.jpg", quality=92)

    # --- Step 3: matching (closest enrolled person, score, decision) -------
    row_h, w, pad, size = 84, 760, 14, 64
    sheet = Image.new("RGB", (w, pad + len(faces) * row_h + 34), "white")
    d = ImageDraw.Draw(sheet)
    bar_x, bar_w = 430, 250
    thr_x = bar_x + int(bar_w * config.MATCH_THRESHOLD)
    for i, (f, res) in enumerate(zip(faces, results)):
        y = pad + i * row_h
        known = res["status"] == "known"
        color = KNOWN if known else UNKNOWN
        best = res["candidates"][0]
        d.text((pad + 8, y + size / 2), str(i + 1), font=font(15, True), fill=color, anchor="mm")
        sheet.paste(to_pil(crop_face(bgr, f.box, size=size)), (pad + 26, y))
        d.text((pad + 26 + size + 14, y + size / 2), "vs", font=font(13), fill=MUTED, anchor="mm")
        sheet.paste(to_pil(cv2.resize(thumbs[best["person_id"]], (size, size))), (pad + 26 + size + 28, y))
        tx = pad + 26 + 2 * size + 42
        d.text((tx, y + 10), best["name"], font=font(15, True), fill=INK)
        d.text((tx, y + 34), "closest person in the folder", font=font(12), fill=MUTED)
        score = max(0.0, min(1.0, best["score"]))
        d.rounded_rectangle((bar_x, y + 14, bar_x + bar_w, y + 26), 6, fill=LINE)
        d.rounded_rectangle((bar_x, y + 14, bar_x + max(8, int(bar_w * score)), y + 26), 6, fill=color)
        d.line((thr_x, y + 8, thr_x, y + 32), fill=INK, width=2)
        verdict = f"{best['score']:.2f}  ->  {res['name'] if known else 'Unknown'}"
        d.text((bar_x, y + 38), verdict, font=font(14, True), fill=color)
        d.line((pad, y + row_h - 10, w - pad, y + row_h - 10), fill=LINE, width=1)
    d.text((thr_x, sheet.height - 16), f"black line = threshold {config.MATCH_THRESHOLD:.2f}", font=font(12),
           fill=MUTED, anchor="mm")
    sheet.save(OUT / "process-3-match.jpg", quality=92)

    print(f"Wrote 3 figures to {OUT} ({len(faces)} faces)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
