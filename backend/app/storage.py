"""Persistence: SQLite for structured data, FAISS for vector search.

SQLite is the source of truth (people, their embeddings, past results).
The FAISS index is a derived, rebuildable search structure over the enrolled
embeddings; it is saved next to the database and rebuilt if it is missing or
out of date.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import faiss
import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS galleries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS people (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    gallery_id  INTEGER NOT NULL REFERENCES galleries(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    file_name   TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    status      TEXT NOT NULL,           -- ok | warning | failed
    note        TEXT,
    faces_found INTEGER NOT NULL DEFAULT 0,
    det_score   REAL,
    embedding   BLOB,                    -- 512 float32, L2-normalised
    thumb       BLOB,                    -- small JPEG of the enrolled face
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_people_gallery ON people(gallery_id);

-- Results of analysing a reference photo, keyed by file content, so reopening
-- the same folder is instant and only new or changed photos are processed.
CREATE TABLE IF NOT EXISTS reference_cache (
    file_hash   TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    note        TEXT,
    faces_found INTEGER NOT NULL,
    det_score   REAL,
    embedding   BLOB,
    thumb       BLOB
);

CREATE TABLE IF NOT EXISTS identifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    gallery_id  INTEGER NOT NULL,
    file_name   TEXT NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    faces_json  TEXT NOT NULL,           -- boxes, detector scores, crops
    embeddings  BLOB NOT NULL,           -- n x 512 float32
    created_at  REAL NOT NULL
);
"""


def _to_blob(vec: np.ndarray | None) -> bytes | None:
    return None if vec is None else np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32).copy()


class Store:
    def __init__(self, db_path: Path = config.DB_PATH, index_dir: Path = config.INDEX_DIR):
        self.db_path = Path(db_path)
        self.index_dir = Path(index_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_lock = threading.Lock()
        self._indexes: dict[int, faiss.Index] = {}
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- galleries ---------------------------------------------------------
    def create_gallery(self, name: str) -> int:
        """Start a new gallery. One folder is active at a time, so older
        galleries (and their indexes and results) are removed."""
        with self._conn() as c:
            old = [r["id"] for r in c.execute("SELECT id FROM galleries")]
            c.execute("DELETE FROM galleries")
            c.execute("DELETE FROM identifications")
            cur = c.execute(
                "INSERT INTO galleries(name, created_at) VALUES (?, ?)", (name, time.time())
            )
            gid = int(cur.lastrowid)
        for g in old:
            self._drop_index(g)
        return gid

    def current_gallery(self) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM galleries ORDER BY id DESC LIMIT 1").fetchone()
            return dict(row) if row else None

    def gallery_exists(self, gallery_id: int) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM galleries WHERE id = ?", (gallery_id,)).fetchone() is not None

    def delete_everything(self) -> None:
        with self._conn() as c:
            old = [r["id"] for r in c.execute("SELECT id FROM galleries")]
            c.execute("DELETE FROM galleries")
            c.execute("DELETE FROM identifications")
            c.execute("DELETE FROM reference_cache")
        for g in old:
            self._drop_index(g)

    # --- people ------------------------------------------------------------
    def add_person(self, gallery_id: int, name: str, file_name: str, file_hash: str, result: dict) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO people(gallery_id, name, file_name, file_hash, status, note,
                                      faces_found, det_score, embedding, thumb, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    gallery_id, name, file_name, file_hash, result["status"], result.get("note"),
                    result.get("faces_found", 0), result.get("det_score"),
                    _to_blob(result.get("embedding")), result.get("thumb"), time.time(),
                ),
            )
            pid = int(cur.lastrowid)
        self._invalidate_index(gallery_id)
        return pid

    def count_people(self, gallery_id: int) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM people WHERE gallery_id = ?", (gallery_id,)).fetchone()[0])

    def list_people(self, gallery_id: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT id, name, file_name, status, note, faces_found, det_score,
                          thumb IS NOT NULL AS has_thumb
                   FROM people WHERE gallery_id = ? ORDER BY name COLLATE NOCASE, id""",
                (gallery_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def person_names(self, gallery_id: int) -> dict[int, str]:
        with self._conn() as c:
            rows = c.execute("SELECT id, name FROM people WHERE gallery_id = ?", (gallery_id,))
            return {int(r["id"]): r["name"] for r in rows}

    def person_thumb(self, person_id: int) -> bytes | None:
        with self._conn() as c:
            row = c.execute("SELECT thumb FROM people WHERE id = ?", (person_id,)).fetchone()
            return row["thumb"] if row else None

    # --- reference cache ---------------------------------------------------
    def cache_get(self, file_hash: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM reference_cache WHERE file_hash = ?", (file_hash,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["embedding"] = _from_blob(out["embedding"])
        return out

    def cache_put(self, file_hash: str, result: dict) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO reference_cache
                   (file_hash, status, note, faces_found, det_score, embedding, thumb)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_hash, result["status"], result.get("note"), result.get("faces_found", 0),
                    result.get("det_score"), _to_blob(result.get("embedding")), result.get("thumb"),
                ),
            )

    # --- FAISS index -------------------------------------------------------
    def _index_path(self, gallery_id: int) -> Path:
        return self.index_dir / f"gallery_{gallery_id}.faiss"

    def _drop_index(self, gallery_id: int) -> None:
        with self._index_lock:
            self._indexes.pop(gallery_id, None)
            self._index_path(gallery_id).unlink(missing_ok=True)

    def _invalidate_index(self, gallery_id: int) -> None:
        with self._index_lock:
            self._indexes.pop(gallery_id, None)

    def _enrolled(self, gallery_id: int) -> tuple[np.ndarray, np.ndarray]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, embedding FROM people WHERE gallery_id = ? AND embedding IS NOT NULL ORDER BY id",
                (gallery_id,),
            ).fetchall()
        ids = np.array([r["id"] for r in rows], dtype=np.int64)
        if not rows:
            return ids, np.zeros((0, config.EMBED_DIM), dtype=np.float32)
        return ids, np.stack([_from_blob(r["embedding"]) for r in rows])

    def build_index(self, gallery_id: int) -> faiss.Index:
        """(Re)build the FAISS index for a gallery from SQLite and save it.

        Embeddings are unit vectors, so inner product == cosine similarity.
        A flat (exact) index is right at this scale; IDs are the people.id keys.
        """
        ids, vecs = self._enrolled(gallery_id)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(config.EMBED_DIM))
        if len(ids):
            index.add_with_ids(vecs, ids)
        # Go through bytes so paths with non-ASCII characters work on Windows.
        self._index_path(gallery_id).write_bytes(faiss.serialize_index(index).tobytes())
        with self._index_lock:
            self._indexes[gallery_id] = index
        return index

    def get_index(self, gallery_id: int) -> faiss.Index:
        with self._index_lock:
            cached = self._indexes.get(gallery_id)
        if cached is not None:
            return cached
        ids, _ = self._enrolled(gallery_id)
        path = self._index_path(gallery_id)
        if path.exists():
            try:
                index = faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8))
                if index.ntotal == len(ids):
                    with self._index_lock:
                        self._indexes[gallery_id] = index
                    return index
            except Exception:
                pass
        return self.build_index(gallery_id)

    # --- identifications ---------------------------------------------------
    def save_identification(self, gallery_id: int, file_name: str, width: int, height: int,
                            faces: list[dict], embeddings: np.ndarray) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO identifications(gallery_id, file_name, width, height,
                                               faces_json, embeddings, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (gallery_id, file_name, width, height, json.dumps(faces), _to_blob(embeddings), time.time()),
            )
            return int(cur.lastrowid)

    def get_identification(self, ident_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM identifications WHERE id = ?", (ident_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["faces"] = json.loads(out.pop("faces_json"))
        out["embeddings"] = _from_blob(out["embeddings"]).reshape(-1, config.EMBED_DIM)
        return out
