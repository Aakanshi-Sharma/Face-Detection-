"""ArcFace face embedder (InsightFace w600k_r50.onnx) on ONNX Runtime.

A face is warped to the canonical 112x112 ArcFace pose using its 5 landmarks,
then mapped to a 512-d unit vector. Same person => high cosine similarity.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Canonical landmark positions in the 112x112 ArcFace crop.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
CROP_SIZE = 112


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama) mapping src -> dst. 2x3."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    rot = u @ np.diag(d) @ vt
    var = (src_c**2).sum() / len(src)
    scale = (s * d).sum() / var if var > 0 else 1.0
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = dst_mean - scale * rot @ src_mean
    return m


def align_face(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    m = similarity_transform(landmarks, ARCFACE_TEMPLATE)
    return cv2.warpAffine(bgr, m, (CROP_SIZE, CROP_SIZE), borderValue=0.0)


class FaceEmbedder:
    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.dynamic_batch = not isinstance(inp.shape[0], int)

    def _run(self, crops: list[np.ndarray]) -> np.ndarray:
        blob = cv2.dnn.blobFromImages(
            crops, 1.0 / 127.5, (CROP_SIZE, CROP_SIZE), (127.5, 127.5, 127.5), swapRB=True
        )
        if self.dynamic_batch:
            return self.session.run(None, {self.input_name: blob})[0]
        return np.concatenate(
            [self.session.run(None, {self.input_name: blob[i : i + 1]})[0] for i in range(len(crops))]
        )

    def embed(self, aligned: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """Embed aligned 112x112 BGR crops -> (n, 512) L2-normalised float32.

        Each crop is embedded together with its mirror image and the two are
        summed: a cheap test-time augmentation that steadies the template.
        """
        if not aligned:
            return np.zeros((0, 512), dtype=np.float32)
        feats = []
        for start in range(0, len(aligned), batch_size):
            chunk = aligned[start : start + batch_size]
            straight = self._run(chunk)
            mirrored = self._run([cv2.flip(c, 1) for c in chunk])
            feats.append(straight + mirrored)
        out = np.concatenate(feats).astype(np.float32)
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out
