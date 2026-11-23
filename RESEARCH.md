# Research

What was looked into before and during the build: the shape of the problem,
the options at each stage, what was picked and rejected, and the measurements
that back the choices.

Two kinds of evidence appear below and are kept apart:

- **Measured here**: numbers produced by `scripts/evaluate.py` on this laptop.
- **From the sources**: properties reported by the linked papers and projects.

## 1. The problem, stated precisely

"Who is in the photo?" is **one-shot, open-set face identification**:

- **One-shot**: there is exactly one reference photo per person.
- **Open-set**: some people in the group photo are not in the folder, and the
  system must say so instead of picking the nearest name.
- **Unseen data**: the judges bring their own folder and photo, so nothing can
  be tuned to specific people.
- **Offline, CPU**: no hosted APIs, and it must run on an ordinary laptop.

These four facts rule out most of the "train a classifier" tutorials and point
to the standard industry pipeline: **detect → align → embed → compare against a
threshold**.

## 2. Existing tools and projects reviewed

| Project | What it is | Verdict |
|---|---|---|
| [InsightFace](https://github.com/deepinsight/insightface) | SCRFD/RetinaFace detectors and ArcFace recognisers, distributed as ONNX model packs | **Used** (model files only, not the Python package) |
| [OpenCV Zoo](https://github.com/opencv/opencv_zoo) | YuNet detector and SFace recogniser, permissive licences, run with OpenCV alone | Kept as the commercial-licence fallback |
| [face_recognition](https://github.com/ageitgey/face_recognition) (dlib) | Very simple API, 128-d embeddings | Rejected: needs a C++ build on Windows; older model, weaker on small or angled faces |
| [DeepFace](https://github.com/serengil/deepface) | Wrapper around many models | Rejected: heavy dependencies (TensorFlow), downloads models at run time by default |
| [facenet-pytorch](https://github.com/timesler/facenet-pytorch) | MTCNN + FaceNet in PyTorch | Rejected: PyTorch is a large install; MTCNN is slow and misses small faces |
| [CompreFace](https://github.com/exadel-inc/CompreFace) | Complete self-hosted face recognition service | Rejected: needs Docker and several services; far more than a laptop app needs |
| [MediaPipe Face Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) | Fast short-range detector | Rejected: designed for selfie-distance faces; no recogniser |

## 3. Options compared at each stage

### Face detection

| Option | Strength | Weakness | Decision |
|---|---|---|---|
| **SCRFD-10GF** ([paper](https://arxiv.org/abs/2105.04714)) | Strong on small faces, fast on CPU, gives 5 landmarks | InsightFace weights are non-commercial | **Picked** |
| RetinaFace ([paper](https://arxiv.org/abs/1905.00641)) | Very accurate | Slower than SCRFD at similar accuracy | Rejected |
| YuNet (OpenCV Zoo) | Tiny, permissive licence | Less accurate on small or hard faces | Fallback |
| MTCNN ([paper](https://arxiv.org/abs/1604.02878)) | Classic, widely used | Slow three-stage cascade, weaker on small faces | Rejected |
| MediaPipe | Very fast | Short-range model | Rejected |

From the sources: SCRFD-10GF reports about 95 / 94 / 83 % AP on the WIDER FACE
easy / medium / hard sets, the hard set being mostly small faces.

### Face embedding

| Option | Strength | Weakness | Decision |
|---|---|---|---|
| **ArcFace ResNet-50**, `w600k_r50` ([paper](https://arxiv.org/abs/1801.07698)) | Trained on about 600k identities; strong with a single reference image | 166 MB, about 0.2 s per face on CPU | **Picked** |
| ArcFace MobileFaceNet (`buffalo_s`) | Several times faster and smaller | Lower accuracy on low-quality faces | Rejected: accuracy matters more than seconds |
| AdaFace ([paper](https://arxiv.org/abs/2204.00964)) | Designed for low-quality images | No ready ONNX pack; would need a PyTorch export step | Rejected for install simplicity |
| FaceNet ([paper](https://arxiv.org/abs/1503.03832)) | Well known | Older; below ArcFace on standard benchmarks | Rejected |
| SFace ([paper](https://arxiv.org/abs/2205.12010)) | Permissive licence, OpenCV only | Lower accuracy | Fallback |
| dlib ResNet | Simple | 128-d, older training data | Rejected |

### Runtime

| Option | Decision |
|---|---|
| **[ONNX Runtime](https://onnxruntime.ai/), CPU** | **Picked**: one small wheel, no GPU, no compiler |
| PyTorch | Rejected: very large install for inference only |
| OpenCV DNN | Rejected: slower than ONNX Runtime for these models |
| `insightface` Python package | Rejected: needs a C++ compiler on Windows. The pre/post-processing was written directly instead (`detector.py`, `embedder.py`) |

### Deciding the name

| Option | Decision |
|---|---|
| Train a classifier (SVM, kNN, fine-tune) on the 50 people | Rejected: one photo per person cannot train it, and a closed-set classifier always answers with one of the 50 names |
| Train on augmented copies of each photo | Rejected: copies of one photo add no new information about the person (see section 5 for how the idea was used instead) |
| Nearest neighbour with a fixed cosine threshold | Baseline |
| **Threshold + one-to-one assignment** ([Hungarian algorithm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)) | **Picked**: a person can appear only once in a photo, so two faces may not share a name |

### Storage and search

| Option | Decision |
|---|---|
| In memory only | Rejected: re-enrols on every launch |
| **SQLite** | **Picked** for people, embeddings, the per-file cache and past results |
| Brute-force NumPy dot product | Would give identical answers at 50 people; not used |
| **[FAISS](https://github.com/facebookresearch/faiss)**, flat inner-product index | **Picked** for vector search: exact results now, and the same code scales to much larger folders |
| [Chroma](https://github.com/chroma-core/chroma), [sqlite-vec](https://github.com/asg017/sqlite-vec) | Rejected: a second database engine for 50 vectors |

### User interface

| Option | Decision |
|---|---|
| Streamlit / Gradio | Rejected: fastest to build, but generic look and little control over drawing on the photo |
| Desktop app (PySide6, Tauri, Electron) | Rejected: native dialogs, but heavier packaging |
| **FastAPI + React (Vite), served locally** | **Picked**: full control of the look, and one command to start |

Folder picking in a browser was the awkward part. Options were the browser's
directory picker, a native dialog opened by the server, or a `pywebview`
wrapper. The browser picker was chosen because it needs nothing extra.

### Securing a local API

A server on `127.0.0.1` can still be called by any web page in the same
browser. The design follows the
[OWASP CSRF guidance](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html)
and uses [Fetch Metadata headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Sec-Fetch-Site):
host allow-list (against DNS rebinding), origin checks with no CORS, an
HttpOnly SameSite=Strict session cookie, and a CSRF token header.

## 4. Data sources

| Source | Use |
|---|---|
| [LFW, Labeled Faces in the Wild](https://scikit-learn.org/stable/datasets/real_world.html#labeled-faces-in-the-wild-dataset) | Practice data: 13k press photos of 5.7k people. Downloaded from the [figshare mirror](https://ndownloader.figshare.com/files/5976015) that scikit-learn uses (the original university site was unreachable) |
| InsightFace `buffalo_l` [model pack](https://github.com/deepinsight/insightface/releases/tag/v0.7) | The two ONNX model files |

The practice set (`scripts/make_practice_data.py`) mirrors the demo: 50 people
enrolled from one photo each, and 12 "group photos" of 10 faces, each made from
a **different** photo of 7 enrolled people plus 3 strangers.

Not used: AI-generated faces. They were allowed for practice data, but real
photos with real variation in pose, age and lighting are a harder and more
honest test.

## 5. Measurements (measured here)

Laptop: Intel Core i5-8250U (4 cores), 8 GB RAM, CPU only.

### Choosing the threshold

Cosine similarity of each group-photo face to people in the folder:

| | Lowest | Typical |
|---|---|---|
| To the **right** person (84 faces) | 0.415 | 0.668 (median) |
| To the closest **wrong** person (120 faces) | 0.229 is the highest seen | 0.131 (median) |

There is a clear gap between 0.23 and 0.41. The threshold sits inside it:

| Threshold | Named correctly | Missed (marked Unknown) | Wrong name | Stranger given a name | Stranger marked Unknown |
|---|---|---|---|---|---|
| 0.20 | 84 | 0 | 0 | **1** | 35 |
| 0.25 | 84 | 0 | 0 | 0 | 36 |
| 0.30 | 84 | 0 | 0 | 0 | 36 |
| **0.35 (default)** | **84** | **0** | **0** | **0** | **36** |
| 0.40 | 84 | 0 | 0 | 0 | 36 |
| 0.45 | 83 | 1 | 0 | 0 | 36 |
| 0.50 | 78 | 6 | 0 | 0 | 36 |

### Robustness to faults

The idea of making damaged copies of photos came up as a way to get training
data. It was not used for training (section 3), but it is exactly the right
tool for testing. The same group photos with a fault applied, at threshold
0.35:

| Fault applied | Named correctly (of 84) | Missed | Wrong name | Stranger given a name | Lowest similarity to the right person |
|---|---|---|---|---|---|
| None | 84 | 0 | 0 | 0 | 0.415 |
| Half resolution | 84 | 0 | 0 | 0 | 0.386 |
| Darkened to 30% | 84 | 0 | 0 | 0 | 0.382 |
| Sensor noise | 81 | 3 | 0 | 0 | 0.304 |
| JPEG quality 12 | 79 | 5 | 0 | 0 | 0.250 |
| Blur (sigma 2) | 77 | 7 | 0 | 0 | 0.267 |

What this shows:

- When quality drops, the system **fails safe**: people become "Unknown". In
  no test was anyone given a wrong name, and no stranger was named.
- Blur and heavy compression hurt most. Resolution and darkness hurt least.
- Lowering the threshold to 0.30 would recover some of the missed faces, at
  the cost of a thinner margin above the highest wrong-person score (0.229).
  The default stays at 0.35 because a wrong name is worse than "Unknown"; the
  slider in the UI lets the user loosen it for a poor photo.

### Speed

| Step | Time |
|---|---|
| Loading both models | about 2 s |
| Enrolling one reference photo | about 0.6 s (30 s for 50), then instant from the cache |
| Analysing a group photo with 10 to 12 faces | about 4 to 6 s |

### Limits of this evidence

- LFW photos are press portraits. The practice "group photos" are collages of
  them, so faces are frontal and evenly sized. A real group photo has smaller,
  turned and partly hidden faces, and will score lower.
- The practice people are public figures photographed years apart, which is
  harder than same-day office photos in one way (age) and easier in another
  (image quality).
- 50 people and 120 test faces is a small sample. The threshold is a sensible
  default, not a guarantee.

## 6. Summary of choices

| Stage | Picked | Main reason |
|---|---|---|
| Detection | SCRFD-10GF | Small faces in group photos, CPU speed |
| Embedding | ArcFace ResNet-50 | Best accuracy from one reference photo |
| Runtime | ONNX Runtime, CPU | Small install, runs on any laptop |
| Decision | Threshold 0.35 + one-to-one assignment | Must be able to say "Unknown" |
| Storage | SQLite + FAISS (flat) | Simple, persistent, exact search |
| UI | FastAPI + React | UI quality is scored |
| Training | None | One photo per person; open-set problem |
