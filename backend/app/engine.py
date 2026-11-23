"""Face engine: detect -> align -> embed, for reference and group photos."""
from __future__ import annotations

import threading

import numpy as np

from . import config
from .detector import Face, FaceDetector, _nms
from .embedder import FaceEmbedder, align_face
from .imaging import crop_face, encode_jpeg


class ModelsMissingError(RuntimeError):
    pass


class FaceEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._detector: FaceDetector | None = None
        self._embedder: FaceEmbedder | None = None

    @staticmethod
    def models_present() -> bool:
        return config.DETECTOR_MODEL.exists() and config.EMBEDDER_MODEL.exists()

    def load(self) -> None:
        with self._lock:
            if self._detector is not None:
                return
            if not self.models_present():
                raise ModelsMissingError(
                    "Model files not found. Run: python scripts/download_models.py"
                )
            self._detector = FaceDetector(
                config.DETECTOR_MODEL, config.DET_SCORE_THRESHOLD, config.DET_NMS_THRESHOLD
            )
            self._embedder = FaceEmbedder(config.EMBEDDER_MODEL)

    @property
    def ready(self) -> bool:
        return self._detector is not None

    # --- reference photo: one person ---------------------------------------
    def analyse_reference(self, bgr: np.ndarray) -> dict:
        """Find the person in a reference photo and compute their template."""
        self.load()
        faces: list[Face] = []
        # 640 suits most portraits; 320 catches faces that fill the frame;
        # 1280 catches a small face in a large photo.
        for det_size in (config.ENROLL_DET_SIZE, 320, 1280):
            faces = self._detector.detect(bgr, det_size)
            if faces:
                break
        if not faces:
            return {"status": "failed", "note": "No face found in this photo.", "faces_found": 0}

        main = max(faces, key=lambda f: f.area)
        embedding = self._embedder.embed([align_face(bgr, main.landmarks)])[0]
        result = {
            "status": "ok",
            "note": None,
            "faces_found": len(faces),
            "det_score": main.score,
            "embedding": embedding,
            "thumb": encode_jpeg(crop_face(bgr, main.box, size=160), quality=85),
        }
        if len(faces) > 1:
            result["status"] = "warning"
            result["note"] = f"{len(faces)} faces in this photo. Used the largest one."
        return result

    # --- group photo: many people ------------------------------------------
    def analyse_group(self, bgr: np.ndarray) -> tuple[list[Face], np.ndarray]:
        """Detect every face in a group photo and embed each one."""
        self.load()
        long_side = max(bgr.shape[:2])
        det_size = int(min(max(long_side, config.ENROLL_DET_SIZE), config.GROUP_DET_SIZE))

        faces = self._detector.detect(bgr, det_size)
        if det_size > config.ENROLL_DET_SIZE:
            # A second, coarser pass picks up very large faces the fine pass can miss.
            faces += self._detector.detect(bgr, config.ENROLL_DET_SIZE)
            if faces:
                boxes = np.stack([f.box for f in faces])
                scores = np.array([f.score for f in faces])
                faces = [faces[i] for i in _nms(boxes, scores, config.DET_NMS_THRESHOLD)]

        faces = [f for f in faces if f.size >= config.MIN_FACE_SIZE]
        faces.sort(key=lambda f: (f.box[0] + f.box[2]) / 2)  # left to right
        embeddings = self._embedder.embed([align_face(bgr, f.landmarks) for f in faces])
        return faces, embeddings
