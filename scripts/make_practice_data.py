"""Build practice data from the public LFW dataset (Labeled Faces in the Wild).

Creates, under practice_data/:

  people/            50 reference photos, one per person, named "<Name>.jpg"
  group_XX.jpg       synthetic "group photos": a collage of ~10 faces, using a
                     DIFFERENT photo of some enrolled people plus strangers
  truth.json         who is really in each collage (None = not in the folder)

This is development data only; the app itself never needs it.
Needs internet once (LFW is ~230 MB). Run with the backend environment:

  uv run --project backend python scripts/make_practice_data.py
"""
from __future__ import annotations

import io
import json
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

URL = "https://ndownloader.figshare.com/files/5976015"  # lfw-funneled.tgz
ROOT = Path(__file__).resolve().parent.parent / "practice_data"
# The 230 MB archive is kept outside the project so it is never synced or committed.
ARCHIVE = Path.home() / ".cache" / "faceid-practice" / "lfw-funneled.tgz"

N_PEOPLE = 50
N_GROUPS = 12
KNOWN_PER_GROUP = 7
STRANGERS_PER_GROUP = 3
TILE = 150  # LFW images are 250px; shrink so faces are group-photo sized


def download() -> None:
    if ARCHIVE.exists():
        return
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARCHIVE.with_suffix(".part")
    print("Downloading LFW (about 230 MB) ...")
    if shutil.which("curl"):
        subprocess.run(["curl", "-L", "--fail", "-o", str(tmp), URL], check=True)
    else:
        with urllib.request.urlopen(URL) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    tmp.rename(ARCHIVE)


def load_image(tf: tarfile.TarFile, member: str) -> Image.Image:
    return Image.open(io.BytesIO(tf.extractfile(member).read())).convert("RGB")


def main() -> int:
    download()
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    print("Reading archive ...")
    tf = tarfile.open(ARCHIVE)
    grouped = defaultdict(list)
    for m in tf.getmembers():
        if m.isfile() and m.name.endswith(".jpg"):
            grouped[Path(m.name).parent.name].append(m.name)
    by_person = {p: sorted(v) for p, v in sorted(grouped.items())}
    multi = [p for p, imgs in by_person.items() if len(imgs) >= 3]
    single = [p for p, imgs in by_person.items() if len(imgs) == 1]
    enrolled = rng.sample(multi, N_PEOPLE)
    strangers = rng.sample(single, N_GROUPS * STRANGERS_PER_GROUP)

    people_dir = ROOT / "people"
    people_dir.mkdir(exist_ok=True)
    for old in people_dir.glob("*.jpg"):
        old.unlink()
    for person in enrolled:
        load_image(tf, by_person[person][0]).save(people_dir / f"{person.replace('_', ' ')}.jpg", quality=95)

    truth = {}
    for g in range(N_GROUPS):
        members = [(p, rng.choice(by_person[p][1:])) for p in rng.sample(enrolled, KNOWN_PER_GROUP)]
        members += [(None, by_person[s][0]) for s in strangers[g * STRANGERS_PER_GROUP:(g + 1) * STRANGERS_PER_GROUP]]
        rng.shuffle(members)
        cols = 5
        rows = -(-len(members) // cols)
        sheet = Image.new("RGB", (cols * TILE, rows * TILE), (30, 30, 30))
        labels = []
        for i, (person, path) in enumerate(members):
            tile = load_image(tf, path).resize((TILE, TILE), Image.LANCZOS)
            sheet.paste(tile, ((i % cols) * TILE, (i // cols) * TILE))
            labels.append({
                "name": person.replace("_", " ") if person else None,
                "center": [(i % cols) * TILE + TILE // 2, (i // cols) * TILE + TILE // 2],
            })
        name = f"group_{g + 1:02d}.jpg"
        sheet.save(ROOT / name, quality=90)
        truth[name] = labels

    (ROOT / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote {N_PEOPLE} reference photos and {N_GROUPS} group collages to {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
