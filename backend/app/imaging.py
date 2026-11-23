"""Image decoding helpers: any common format in, upright BGR array out."""
from __future__ import annotations

import base64
import io
import re
from pathlib import PurePath

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # HEIC/HEIF photos from phones
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional
    pass

# Guard against decompression bombs but allow big camera photos.
Image.MAX_IMAGE_PIXELS = 200_000_000


class ImageDecodeError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes to a BGR uint8 array with EXIF rotation applied."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            rgb = np.asarray(im)
    except Exception as exc:
        raise ImageDecodeError("This file could not be opened as a photo.") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_jpeg(bgr: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def to_data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def crop_face(bgr: np.ndarray, box, size: int = 128, margin: float = 0.35) -> np.ndarray:
    """Square crop around a face box with some margin, resized to `size`."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + margin) / 2
    left, top = int(round(cx - half)), int(round(cy - half))
    side = max(int(round(half * 2)), 2)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    sx1, sy1 = max(left, 0), max(top, 0)
    sx2, sy2 = min(left + side, w), min(top + side, h)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - top : sy2 - top, sx1 - left : sx2 - left] = bgr[sy1:sy2, sx1:sx2]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interp)


def name_from_filename(filename: str) -> str:
    """'photos/priya_sharma.JPG' -> 'priya sharma'. The file name is the label."""
    stem = PurePath(filename.replace("\\", "/")).stem
    stem = re.sub(r"[_]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or "Unnamed"
