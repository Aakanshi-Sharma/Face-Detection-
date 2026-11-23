"""SCRFD face detector (InsightFace det_10g.onnx) on ONNX Runtime.

Implemented directly on onnxruntime so the app has no dependency on the
`insightface` Python package (which needs a C++ compiler on Windows).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

STRIDES = (8, 16, 32)
ANCHORS_PER_CELL = 2


@dataclass
class Face:
    box: np.ndarray        # [x1, y1, x2, y2] in original image pixels
    landmarks: np.ndarray  # 5 x 2: eyes, nose, mouth corners
    score: float           # detector confidence 0..1

    @property
    def size(self) -> float:
        return float(min(self.box[2] - self.box[0], self.box[3] - self.box[1]))

    @property
    def area(self) -> float:
        return float((self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= threshold]
    return keep


class FaceDetector:
    def __init__(self, model_path: Path, score_threshold: float = 0.5, nms_threshold: float = 0.4):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            grid = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            centers = np.stack([centers] * ANCHORS_PER_CELL, axis=1).reshape(-1, 2)
            self._centers[key] = centers
        return self._centers[key]

    def detect(self, bgr: np.ndarray, det_size: int = 640) -> list[Face]:
        """Detect faces. `det_size` is the long side the image is scaled to."""
        ih, iw = bgr.shape[:2]
        scale = det_size / max(ih, iw)
        nw, nh = max(int(round(iw * scale)), 1), max(int(round(ih * scale)), 1)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(bgr, (nw, nh), interpolation=interp)

        # Pad to a multiple of 32 (the coarsest stride); image stays top-left.
        ph, pw = -(-nh // 32) * 32, -(-nw // 32) * 32
        canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (pw, ph), (127.5, 127.5, 127.5), swapRB=True
        )
        outputs = self.session.run(None, {self.input_name: blob})

        # Group outputs by what they are (last dim) rather than trusting order.
        scores_by_n, boxes_by_n, kps_by_n = {}, {}, {}
        for out in outputs:
            arr = out.reshape(-1, out.shape[-1])
            target = {1: scores_by_n, 4: boxes_by_n, 10: kps_by_n}.get(arr.shape[1])
            if target is not None:
                target[arr.shape[0]] = arr

        all_boxes, all_scores, all_kps = [], [], []
        for stride in STRIDES:
            fh, fw = ph // stride, pw // stride
            n = fh * fw * ANCHORS_PER_CELL
            if n not in scores_by_n:
                continue
            scores = scores_by_n[n][:, 0]
            keep = np.where(scores >= self.score_threshold)[0]
            if keep.size == 0:
                continue
            centers = self._anchor_centers(fh, fw, stride)[keep]
            dist = boxes_by_n[n][keep] * stride
            boxes = np.stack(
                [
                    centers[:, 0] - dist[:, 0],
                    centers[:, 1] - dist[:, 1],
                    centers[:, 0] + dist[:, 2],
                    centers[:, 1] + dist[:, 3],
                ],
                axis=1,
            )
            kps = kps_by_n[n][keep].reshape(-1, 5, 2) * stride + centers[:, None, :]
            all_boxes.append(boxes)
            all_scores.append(scores[keep])
            all_kps.append(kps)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes) / scale
        scores = np.concatenate(all_scores)
        kps = np.concatenate(all_kps) / scale

        faces = []
        for i in _nms(boxes, scores, self.nms_threshold):
            box = boxes[i].copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, iw - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, ih - 1)
            faces.append(Face(box=box, landmarks=kps[i], score=float(scores[i])))
        return faces
