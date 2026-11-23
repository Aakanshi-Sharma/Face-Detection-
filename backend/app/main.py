"""FastAPI app: local-only API plus the built React UI.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import hashlib
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, matching, security
from .engine import FaceEngine, ModelsMissingError
from .imaging import ImageDecodeError, crop_face, decode_image, encode_jpeg, name_from_filename, to_data_url
from .storage import Store

import cv2
from pathlib import PurePath

engine = FaceEngine()
store = Store()
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_MAX_SIDE = 2000
KEEP_UPLOADS = 20

# Face inference is CPU-heavy; cap how many run at once.
_inference = threading.BoundedSemaphore(config.MAX_CONCURRENT_INFERENCE)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the models in the background so the UI opens immediately.
    def warm():
        try:
            engine.load()
        except ModelsMissingError:
            pass

    threading.Thread(target=warm, daemon=True).start()
    yield


# No interactive docs or schema endpoint: nothing to explore from outside the UI.
app = FastAPI(title="Who is in the photo?", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(security.SecurityMiddleware)


def _read_upload(file: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing anything over `limit` bytes."""
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"That file is too large (limit {limit // (1024 * 1024)} MB).")
    if not data:
        raise HTTPException(400, "That file is empty.")
    return data


def _clean_filename(filename: str) -> str:
    """Keep only the base name; reject control characters and silly lengths."""
    base = PurePath(filename.replace("\\", "/")).name.strip()
    if not base or len(base) > config.MAX_FILENAME_CHARS or any(ord(ch) < 32 for ch in base):
        raise HTTPException(400, "That file name is not valid.")
    if PurePath(base).suffix.lower() not in config.IMAGE_EXTENSIONS:
        raise HTTPException(415, "That file is not a supported photo type.")
    return base


def _gallery_payload() -> dict | None:
    g = store.current_gallery()
    if not g:
        return None
    people = store.list_people(g["id"])
    return {
        "id": g["id"],
        "name": g["name"],
        "people": people,
        "enrolled": sum(1 for p in people if p["status"] != "failed"),
        "failed": sum(1 for p in people if p["status"] == "failed"),
    }


def _require_engine() -> None:
    try:
        engine.load()
    except ModelsMissingError as exc:
        raise HTTPException(503, str(exc)) from exc


# --- session ---------------------------------------------------------------
@app.get("/api/session")
def session(response: Response):
    """Called once by the page: sets the session cookie, returns the CSRF token."""
    return security.start_session(response)


# --- status ----------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "models_present": engine.models_present(),
        "ready": engine.ready,
        "threshold": config.MATCH_THRESHOLD,
    }


# --- gallery (the folder of known people) ----------------------------------
class GalleryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


@app.get("/api/gallery")
def get_gallery():
    return {"gallery": _gallery_payload()}


@app.post("/api/gallery")
def create_gallery(body: GalleryCreate):
    gid = store.create_gallery(body.name.strip())
    for f in config.UPLOAD_DIR.glob("*.jpg"):
        f.unlink(missing_ok=True)
    return {"id": gid, "name": body.name.strip()}


@app.post("/api/gallery/{gallery_id}/people")
def add_person(gallery_id: int, file: UploadFile = File(...), filename: str = Form(...)):
    """Enrol one reference photo. The file name (without extension) is the name."""
    if not store.gallery_exists(gallery_id):
        raise HTTPException(404, "This folder is no longer open. Choose the folder again.")
    filename = _clean_filename(filename)
    if store.count_people(gallery_id) >= config.MAX_PEOPLE:
        raise HTTPException(400, f"A folder can hold at most {config.MAX_PEOPLE} people.")
    data = _read_upload(file, config.MAX_REFERENCE_BYTES)
    file_hash = hashlib.sha256(data).hexdigest()
    name = name_from_filename(filename)

    result = store.cache_get(file_hash)
    cached = result is not None
    if result is None:
        _require_engine()
        try:
            with _inference:
                result = engine.analyse_reference(decode_image(data))
        except ImageDecodeError as exc:
            result = {"status": "failed", "note": str(exc), "faces_found": 0}
        store.cache_put(file_hash, result)

    pid = store.add_person(gallery_id, name, filename, file_hash, result)
    return {
        "id": pid,
        "name": name,
        "file_name": filename,
        "status": result["status"],
        "note": result.get("note"),
        "faces_found": result.get("faces_found", 0),
        "has_thumb": result.get("thumb") is not None,
        "cached": cached,
    }


@app.post("/api/gallery/{gallery_id}/finalize")
def finalize_gallery(gallery_id: int):
    """Build the FAISS search index once all photos are enrolled."""
    if not store.gallery_exists(gallery_id):
        raise HTTPException(404, "This folder is no longer open. Choose the folder again.")
    index = store.build_index(gallery_id)
    payload = _gallery_payload()
    payload["indexed"] = index.ntotal
    return {"gallery": payload}


@app.delete("/api/gallery")
def delete_gallery():
    """Forget everything: people, cached templates, results and uploads."""
    store.delete_everything()
    for f in config.UPLOAD_DIR.glob("*.jpg"):
        f.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/people/{person_id}/thumb")
def person_thumb(person_id: int):
    thumb = store.person_thumb(person_id)
    if not thumb:
        raise HTTPException(404, "No thumbnail")
    return Response(thumb, media_type="image/jpeg")


# --- identification (the group photo) --------------------------------------
def _identify_payload(ident: dict, threshold: float) -> dict:
    gid = ident["gallery_id"]
    index = store.get_index(gid)
    person_ids, sims = matching.similarity_matrix(index, ident["embeddings"])
    results = matching.assign(person_ids, sims, store.person_names(gid), threshold=threshold)
    faces = [{**face, **res, "index": i} for i, (face, res) in enumerate(zip(ident["faces"], results))]
    known = sum(1 for f in faces if f["status"] == "known")
    return {
        "id": ident["id"],
        "file_name": ident["file_name"],
        "width": ident["width"],
        "height": ident["height"],
        "image_url": f"/api/identifications/{ident['id']}/image",
        "threshold": threshold,
        "faces": faces,
        "summary": {"total": len(faces), "known": known, "unknown": len(faces) - known},
    }


@app.post("/api/identify")
def identify(file: UploadFile = File(...), threshold: float | None = Form(None)):
    gallery = store.current_gallery()
    if not gallery:
        raise HTTPException(400, "Choose the folder of people first.")
    _require_engine()
    started = time.perf_counter()
    if threshold is not None and not 0.05 <= threshold <= 0.95:
        raise HTTPException(422, "Threshold must be between 0.05 and 0.95.")
    file_name = _clean_filename(file.filename or "photo.jpg")
    try:
        bgr = decode_image(_read_upload(file, config.MAX_GROUP_BYTES))
    except ImageDecodeError as exc:
        raise HTTPException(400, str(exc)) from exc

    with _inference:
        faces, embeddings = engine.analyse_group(bgr)
    h, w = bgr.shape[:2]
    face_rows = [
        {
            "box": [round(float(v), 1) for v in f.box],
            "det_score": round(f.score, 3),
            "size": round(f.size),
            "crop": to_data_url(encode_jpeg(crop_face(bgr, f.box, size=112), quality=82)),
        }
        for f in faces
    ]
    ident_id = store.save_identification(gallery["id"], file_name, w, h, face_rows, embeddings)

    # Keep an upright, reasonably sized copy for the UI to display.
    scale = min(1.0, DISPLAY_MAX_SIDE / max(h, w))
    display = bgr if scale == 1.0 else cv2.resize(
        bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
    )
    (config.UPLOAD_DIR / f"{ident_id}.jpg").write_bytes(encode_jpeg(display, quality=90))
    old = sorted(config.UPLOAD_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)[:-KEEP_UPLOADS]
    for f in old:
        f.unlink(missing_ok=True)

    payload = _identify_payload(store.get_identification(ident_id), threshold or config.MATCH_THRESHOLD)
    payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    return payload


class Rematch(BaseModel):
    threshold: float = Field(ge=0.05, le=0.95)


@app.post("/api/identifications/{ident_id}/rematch")
def rematch(ident_id: int, body: Rematch):
    """Re-decide names with a different strictness. No model inference needed."""
    ident = store.get_identification(ident_id)
    if not ident:
        raise HTTPException(404, "Result not found")
    return _identify_payload(ident, body.threshold)


@app.get("/api/identifications/{ident_id}/image")
def identification_image(ident_id: int):
    path = config.UPLOAD_DIR / f"{ident_id}.jpg"
    if store.get_identification(ident_id) is None or not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/jpeg")


# --- built frontend --------------------------------------------------------
if config.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = (config.FRONTEND_DIST / path).resolve()
        if path and target.is_file() and config.FRONTEND_DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(config.FRONTEND_DIST / "index.html")
