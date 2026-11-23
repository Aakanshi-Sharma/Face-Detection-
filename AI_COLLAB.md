# AI collaboration

How this project was built with an AI coding assistant (Claude Code), what it
suggested, and where I steered it. The full, unedited chat history is in
[`docs/chat-history/`](docs/chat-history/), starting from my first prompt.

## How we worked

I did not let the AI start coding straight away. My first prompt asked it to
understand the task and lay out the options for each stage; building started
only after I approved a stack. After the first working version I reviewed it
and sent it back twice: once for security and motion, once for the documents.

| Phase | What I asked for | What came back |
|---|---|---|
| 1. Understand | Read the task, explain what is needed, list methods per stage | A brief with option tables and five decisions for me to make |
| 2. Challenge | My own idea: augmented copies as training data | A disagreement with reasons, and a different use for the idea |
| 3. Approve | The stack, with my own choice for vector search | The first working build, tested on practice data |
| 4. Review | More animation, a secured API | A security gateway with tests, and UI motion |
| 5. Deliver | The four documents, with images in the README | These documents |

## Decisions

### Decision 1: plan first, build after approval

- **AI suggested:** nothing yet; this was my instruction in the first prompt.
- **I decided:** the AI must first explain the problem and the methods
  available at each stage (design and model, UI, storage, integration), and
  wait for my approval of the tech stack before writing code.
- **Why:** I wanted to choose between real alternatives, not review a finished
  build I had no say in.
- **Evidence:** chat excerpt #1.

### Decision 2: which face models

- **AI suggested:** SCRFD for detection and ArcFace for embeddings on ONNX
  Runtime, compared against YuNet, MTCNN, MediaPipe, FaceNet, SFace and dlib.
  It also flagged that the InsightFace weights are licensed for non-commercial
  research and named YuNet + SFace as the permissive fallback.
- **I decided:** SCRFD + ArcFace on ONNX Runtime.
- **Why:** the demo is a group photo with small faces and only one reference
  photo per person, so detection of small faces and one-shot accuracy matter
  most. The licence limit is acceptable for a take-home and is documented.
- **Evidence:** chat excerpt #3; accuracy numbers in `RESEARCH.md`.

### Decision 3: no training; my augmentation idea used differently

- **I proposed:** making several copies of each folder photo with different
  faults, to get more training data.
- **AI suggested:** not to train at all. Its reasons: degraded copies of one
  photo add no new information about the person; the pretrained model already
  handles blur and lighting; a classifier over 50 names cannot answer
  "unknown"; and training on the judges' folder live would take minutes.
- **I decided:** accept that. No training step. My idea is used in two other
  places instead: each enrolled face is embedded together with its mirror
  image, and `scripts/evaluate.py` applies faults (blur, half resolution, heavy
  JPEG, darkness, noise) to the practice photos to test robustness and check
  the threshold.
- **Why:** the demo includes strangers, so being able to say "unknown" matters
  more than squeezing the 50 known people.
- **Evidence:** chat excerpt #2; robustness table in `RESEARCH.md`.

### Decision 4: FAISS for vector search (against the AI's suggestion)

- **AI suggested:** SQLite plus a brute-force NumPy comparison. It called a
  vector database "overkill for 50 vectors".
- **I decided:** SQLite for structured data and FAISS for the vectors.
- **Why:** _TODO (my words): add my reason here, for example scaling beyond 50
  people or wanting a proper vector index in the design._
- **Result:** the AI implemented it with SQLite as the source of truth and an
  exact (flat) FAISS index that is rebuilt from SQLite, so my choice costs no
  accuracy. The trade-off is recorded in `ARCHITECTURE.md`.
- **Evidence:** chat excerpt #3.

### Decision 5: FastAPI + React instead of a quick UI

- **AI suggested:** three options: Streamlit/Gradio (fastest, generic look),
  a desktop app (native dialogs, heavier packaging), or FastAPI + React
  (most control, most code). It recommended the last.
- **I decided:** FastAPI + React.
- **Why:** the UI is part of the score, and the users are office staff, so
  the screen has to be one clear flow that looks good.
- **Evidence:** chat excerpt #3; screenshots in `README.md`.

### Decision 6: secure the API, even though it is local

- **AI's first build:** bound the server to `127.0.0.1` and stopped there. The
  endpoints had no authentication.
- **I decided:** the gateway needed to be more secure, with secured endpoints.
- **Why:** _TODO (my words): confirm or replace._ A local server can still be
  called by other web pages open in the same browser, and the data is faces
  and names.
- **Result:** one gateway in front of every endpoint: host allow-list, origin
  checks, an HttpOnly session cookie, a CSRF token on every change, upload
  limits, rate limits, security headers, and the API docs switched off. 16
  security tests were added.
- **Evidence:** chat excerpt #4; `backend/tests/test_security.py`.

### Decision 7: a more animated UI

- **AI's first build:** a static UI with only a progress bar and a scan line.
- **I decided:** make it more animated.
- **Why:** UI/UX is judged, and motion makes the two-step flow easier to
  follow (faces "lock on" one by one, then the list fills in).
- **Result:** entrance, loading and result animations. The AI added, on its
  own, that all motion switches off for users who ask their system to reduce
  motion; I kept that.
- **Evidence:** chat excerpt #4.

### Decision 8: images in the README

- **AI's first README:** text only.
- **I decided:** add images of the UI and of the process.
- **Why:** a reviewer should understand the app without installing it.
- **Result:** six UI screenshots, three figures generated from real pipeline
  output (`scripts/make_readme_figures.py`), and a flow diagram.
- **Evidence:** `README.md`, `docs/images/`.

### Decision 9: an AI choice I reviewed and kept

- **AI decided during the build:** not to use the `insightface` Python
  package, and to write the SCRFD and ArcFace pre/post-processing directly on
  ONNX Runtime, because the package needs a C++ compiler on Windows.
- **I decided:** keep it.
- **Why:** install must be simple on the demo laptop. The cost, about 250
  lines of code to maintain, is noted in `ARCHITECTURE.md`.
- **Evidence:** `backend/app/detector.py`, `backend/app/embedder.py`.

## Where the AI was honest about limits

Things it told me were not verified, which I should check myself before the
demo:

- The practice photos are collages of portraits (LFW), which is easier than a
  real group photo with small, angled or half-hidden faces.
- The browser's real folder dialog, the two Save buttons and HEIC photos were
  not exercised in its tests.
- Animations were checked by screenshot, not watched in motion.

## Chat excerpts

Pasted from the chat history. My messages are verbatim; AI replies are
shortened where marked `[...]`.

### Chat excerpt #1: plan first

> **me:** Here is the task file , we need to build the complete application
> for face recognition , before this first understand what is needed , Firstly
> system design and architecture and model selection , secondly , a UI/UX
> design ,third storage and retrievals and last the complete connection between
> all modules . Give a brief idea to do this task and what are the various
> method which can be done at different stages , After approval the tech stack
> , we will start building the application
>
> **AI:** [...] Technically this is one-shot, open-set face identification: one
> reference photo per person, and the system must refuse to name strangers. The
> "unknown" handling and small faces in a group shot are where the demo is most
> likely to go wrong. [...]
> `AI_COLLAB.md` is scored on decisions you made, so these are yours to confirm
> or change: 1. Models [...] 2. UI [...] 3. Folder picking [...] 4. Storage
> [...] 5. Location [...]

### Chat excerpt #2: my augmentation idea

> **me:** for training purpose , what if we make multiple copied with different
> fault for each photo present in folder , which will give us more training
> data ?
>
> **AI:** Augmentation helps here, but not as training data: the recommended
> design has no training step, and adding one would hurt. [...] A classifier
> over 50 names must pick one of the 50 for every face, so strangers get
> confidently mislabelled. [...] So I'd keep the pretrained model with no
> training, and use your augmentation idea in those three places [enrolment
> template, threshold calibration, robustness testing].

### Chat excerpt #3: approving the stack, and choosing FAISS

> **AI (earlier, on storage):** Vector database (FAISS, Chroma): Overkill for
> 50 vectors, where a brute-force NumPy comparison takes microseconds. [...] I
> recommend SQLite for people, embeddings and past results, with brute-force
> cosine search.
>
> **me:** approved, go with the recommended stack which is Models: SCRFD +
> ArcFace on ONNX Runtime,UI: FastAPI + React , for storage go for sqlite for
> structured and for vector database , use FAISS

### Chat excerpt #4: sending the first build back

> **me:** Make the UI more animated and gateway more secure with secure
> endpoints
>
> **AI:** [...] Every request now passes through one gateway [...] Two limits
> remain [...] Plain HTTP: traffic is unencrypted, but it stays on the loopback
> interface and never leaves the machine. Local file access: another program
> running under your Windows account can read the `data/` folder directly.
