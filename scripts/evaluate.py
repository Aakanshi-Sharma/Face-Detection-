"""Measure accuracy on the practice data and help choose the match threshold.

  uv run --project backend python scripts/evaluate.py

For every face in the practice group photos we know the truth (a name, or
"not in the folder"), so we can count:

  correct        enrolled person given the right name
  missed         enrolled person marked Unknown
  wrong name     enrolled person given someone else's name
  false accept   stranger given a name            (the worst mistake)
  correct unk.   stranger marked Unknown
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import faiss  # noqa: E402

from app import config, matching  # noqa: E402
from app.engine import FaceEngine  # noqa: E402
from app.imaging import decode_image, name_from_filename  # noqa: E402

DATA = ROOT / "practice_data"


def main() -> int:
    truth = json.loads((DATA / "truth.json").read_text())
    engine = FaceEngine()
    engine.load()

    # Enrol the folder.
    t0 = time.perf_counter()
    names, vecs = {}, []
    for i, path in enumerate(sorted((DATA / "people").iterdir())):
        res = engine.analyse_reference(decode_image(path.read_bytes()))
        if res["status"] == "failed":
            print("  could not enrol", path.name)
            continue
        names[i] = name_from_filename(path.name)
        vecs.append((i, res["embedding"]))
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(config.EMBED_DIM))
    index.add_with_ids(np.stack([v for _, v in vecs]), np.array([i for i, _ in vecs], dtype=np.int64))
    enrol_s = time.perf_counter() - t0
    print(f"Enrolled {len(vecs)} people in {enrol_s:.1f}s ({enrol_s / len(vecs):.2f}s each)")

    # Analyse each group photo once; matching is then re-run per threshold.
    photos = []
    t0 = time.perf_counter()
    for name, labels in truth.items():
        faces, emb = engine.analyse_group(decode_image((DATA / name).read_bytes()))
        person_ids, sims = matching.similarity_matrix(index, emb)
        # Which detected face is each labelled person? (label centre inside box)
        owners = []
        for lab in labels:
            cx, cy = lab["center"]
            hit = [k for k, f in enumerate(faces) if f.box[0] <= cx <= f.box[2] and f.box[1] <= cy <= f.box[3]]
            owners.append(hit[0] if hit else None)
        photos.append((name, labels, owners, person_ids, sims, len(faces)))
    per_photo = (time.perf_counter() - t0) / len(photos)
    print(f"Analysed {len(photos)} group photos, {per_photo:.1f}s each\n")

    genuine, impostor = [], []
    for _, labels, owners, person_ids, sims, _ in photos:
        name_to_col = {names[int(p)]: j for j, p in enumerate(person_ids)}
        for lab, k in zip(labels, owners):
            if k is None:
                continue
            if lab["name"]:
                genuine.append(sims[k, name_to_col[lab["name"]]])
                impostor.append(np.delete(sims[k], name_to_col[lab["name"]]).max())
            else:
                impostor.append(sims[k].max())
    genuine, impostor = np.array(genuine), np.array(impostor)
    print("Similarity to the TRUE person   : min %.3f  5th pct %.3f  median %.3f" % (
        genuine.min(), np.percentile(genuine, 5), np.median(genuine)))
    print("Best similarity to a WRONG person: max %.3f  95th pct %.3f  median %.3f\n" % (
        impostor.max(), np.percentile(impostor, 95), np.median(impostor)))

    print("threshold  correct  missed  wrong-name  false-accept  correct-unknown  not-detected")
    for thr in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        c = dict(correct=0, missed=0, wrong=0, false_accept=0, correct_unknown=0, undetected=0)
        for _, labels, owners, person_ids, sims, _ in photos:
            results = matching.assign(person_ids, sims, names, threshold=thr)
            for lab, k in zip(labels, owners):
                if k is None:
                    c["undetected"] += 1
                    continue
                got = results[k]["name"]
                if lab["name"]:
                    c["correct" if got == lab["name"] else "missed" if got is None else "wrong"] += 1
                else:
                    c["correct_unknown" if got is None else "false_accept"] += 1
        mark = "  <- default" if abs(thr - config.MATCH_THRESHOLD) < 1e-9 else ""
        print(f"  {thr:.2f}     {c['correct']:5d}   {c['missed']:5d}   {c['wrong']:8d}   {c['false_accept']:10d}   "
              f"{c['correct_unknown']:13d}   {c['undetected']:10d}{mark}")

    # --- robustness: the same photos with common faults applied --------------
    # (Idea from the project owner: damage the photos on purpose and see what
    # still works. Used here for testing and calibration, not for training.)
    import cv2

    def blur(img): return cv2.GaussianBlur(img, (0, 0), 2.0)
    def low_res(img):
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    def jpeg(img): return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 12])[1], 1)
    def dark(img): return (img.astype(np.float32) * 0.3).astype(np.uint8)
    def noise(img):
        rng = np.random.default_rng(0)
        return np.clip(img.astype(np.float32) + rng.normal(0, 22, img.shape), 0, 255).astype(np.uint8)

    print()
    print("Robustness at the default threshold (%.2f):" % config.MATCH_THRESHOLD)
    print("fault            correct  missed  wrong-name  false-accept  correct-unknown  not-detected  lowest-true-sim")
    for label, fn in [("blur (sigma 2)", blur), ("half resolution", low_res), ("JPEG quality 12", jpeg),
                      ("dark (30%)", dark), ("sensor noise", noise)]:
        c = dict(correct=0, missed=0, wrong=0, false_accept=0, correct_unknown=0, undetected=0)
        lowest = 1.0
        for name, labels in truth.items():
            img = fn(decode_image((DATA / name).read_bytes()))
            faces, emb = engine.analyse_group(img)
            person_ids, sims = matching.similarity_matrix(index, emb)
            results = matching.assign(person_ids, sims, names)
            name_to_col = {names[int(p)]: j for j, p in enumerate(person_ids)}
            for lab in labels:
                cx, cy = lab["center"]
                hit = [k for k, f in enumerate(faces) if f.box[0] <= cx <= f.box[2] and f.box[1] <= cy <= f.box[3]]
                if not hit:
                    c["undetected"] += 1
                    continue
                got = results[hit[0]]["name"]
                if lab["name"]:
                    lowest = min(lowest, float(sims[hit[0], name_to_col[lab["name"]]]))
                    c["correct" if got == lab["name"] else "missed" if got is None else "wrong"] += 1
                else:
                    c["correct_unknown" if got is None else "false_accept"] += 1
        print(f"  {label:15s} {c['correct']:6d}  {c['missed']:6d}  {c['wrong']:10d}  {c['false_accept']:12d}  "
              f"{c['correct_unknown']:15d}  {c['undetected']:12d}  {lowest:15.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
