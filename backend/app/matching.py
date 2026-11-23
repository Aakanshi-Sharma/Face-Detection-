"""Open-set matching: give each face a name, or 'unknown'.

Two rules decide the outcome:

1. Threshold. A face is only named if its cosine similarity to an enrolled
   person is at least `threshold`. Everyone else is "unknown". This is what
   stops strangers being given a name.
2. One-to-one. A person can appear only once in a photo, so names are assigned
   with the Hungarian algorithm: if two faces both look most like Priya, the
   better match keeps the name and the other is re-considered.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import config


def similarity_matrix(index, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of every query face to every enrolled person.

    Returns (person_ids [m], sims [n, m]) using a full FAISS search.
    """
    m = index.ntotal
    n = len(queries)
    if m == 0 or n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((n, 0), dtype=np.float32)
    scores, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), m)
    person_ids = np.sort(ids[0])
    col = {int(pid): j for j, pid in enumerate(person_ids)}
    sims = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for score, pid in zip(scores[i], ids[i]):
            sims[i, col[int(pid)]] = score
    return person_ids, sims


def assign(person_ids: np.ndarray, sims: np.ndarray, names: dict[int, str],
           threshold: float = config.MATCH_THRESHOLD,
           hint_threshold: float = config.HINT_THRESHOLD, top_k: int = 3) -> list[dict]:
    """Decide a result for each face (row of `sims`)."""
    n, m = sims.shape
    matched: dict[int, int] = {}
    if n and m:
        # Only similarities above the threshold are worth anything.
        gain = np.where(sims >= threshold, sims, 0.0)
        rows, cols = linear_sum_assignment(gain, maximize=True)
        matched = {int(r): int(c) for r, c in zip(rows, cols) if sims[r, c] >= threshold}

    results = []
    for i in range(n):
        order = np.argsort(-sims[i])[:top_k] if m else []
        candidates = [
            {
                "person_id": int(person_ids[j]),
                "name": names.get(int(person_ids[j]), "?"),
                "score": round(float(sims[i, j]), 4),
            }
            for j in order
        ]
        if i in matched:
            j = matched[i]
            pid = int(person_ids[j])
            results.append({
                "status": "known",
                "person_id": pid,
                "name": names.get(pid, "?"),
                "score": round(float(sims[i, j]), 4),
                "candidates": candidates,
            })
        else:
            best = candidates[0] if candidates else None
            hint = best if best and best["score"] >= hint_threshold else None
            results.append({
                "status": "unknown",
                "person_id": None,
                "name": None,
                "score": best["score"] if best else 0.0,
                "hint": hint,
                "candidates": candidates,
            })
    return results
