"""Matching rules: threshold decides unknown, one name per person per photo."""
import faiss
import numpy as np

from app import matching

NAMES = {10: "Asha", 20: "Ben", 30: "Chen"}


def unit(*v):
    a = np.zeros(512, dtype=np.float32)
    a[: len(v)] = v
    return a / np.linalg.norm(a)


def build_index():
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(512))
    vecs = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])
    index.add_with_ids(vecs, np.array([10, 20, 30], dtype=np.int64))
    return index


def run(queries, threshold=0.35):
    ids, sims = matching.similarity_matrix(build_index(), np.stack(queries))
    return matching.assign(ids, sims, NAMES, threshold=threshold)


def test_clear_match_is_named():
    (res,) = run([unit(1, 0.1, 0)])
    assert res["status"] == "known" and res["name"] == "Asha"


def test_stranger_is_unknown():
    stranger = np.zeros(512, dtype=np.float32)
    stranger[100] = 1.0
    (res,) = run([stranger])
    assert res["status"] == "unknown" and res["name"] is None


def test_below_threshold_is_unknown_with_hint():
    (res,) = run([unit(0.3, 0, 0, 1)])  # similarity to Asha ~0.29
    assert res["status"] == "unknown"
    assert res["hint"]["name"] == "Asha"


def test_one_name_per_person():
    # Both faces look most like Asha; only the better one may take the name.
    a, b = run([unit(1, 0.05, 0), unit(1, 0.6, 0)])
    assert a["name"] == "Asha"
    assert b["name"] == "Ben"  # falls back to its next-best match above threshold


def test_second_face_unknown_when_no_other_match():
    a, b = run([unit(1, 0, 0), unit(1, 0.1, 0, 0.5)])
    assert [a["name"], b["name"]].count("Asha") == 1
    assert "unknown" in (a["status"], b["status"])


def test_empty_gallery():
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(512))
    ids, sims = matching.similarity_matrix(index, np.stack([unit(1, 0, 0)]))
    (res,) = matching.assign(ids, sims, {})
    assert res["status"] == "unknown"
