// Typed client for the local API. Every request goes to this machine only.

export type PersonStatus = "ok" | "warning" | "failed";

export interface Person {
  id: number;
  name: string;
  file_name: string;
  status: PersonStatus;
  note: string | null;
  faces_found: number;
  has_thumb: boolean | number;
}

export interface Gallery {
  id: number;
  name: string;
  people: Person[];
  enrolled: number;
  failed: number;
}

export interface Candidate {
  person_id: number;
  name: string;
  score: number;
}

export interface FaceResult {
  index: number;
  box: [number, number, number, number];
  det_score: number;
  size: number;
  crop: string;
  status: "known" | "unknown";
  person_id: number | null;
  name: string | null;
  score: number;
  hint?: Candidate | null;
  candidates: Candidate[];
}

export interface Identification {
  id: number;
  file_name: string;
  width: number;
  height: number;
  image_url: string;
  threshold: number;
  faces: FaceResult[];
  summary: { total: number; known: number; unknown: number };
  elapsed_ms?: number;
}

export interface Health {
  models_present: boolean;
  ready: boolean;
  threshold: number;
}

// --- session ---------------------------------------------------------------
// The API only answers our own page. On first use the page asks for a session:
// the server sets an HttpOnly cookie and returns a token that must accompany
// every request that changes something.
let csrfToken: string | null = null;
let sessionPromise: Promise<void> | null = null;

function ensureSession(force = false): Promise<void> {
  if (force) sessionPromise = null;
  if (!sessionPromise) {
    sessionPromise = fetch("/api/session", { credentials: "same-origin" })
      .then((res) => {
        if (!res.ok) throw new Error("Could not start a secure session with the app.");
        return res.json();
      })
      .then((body: { csrf_token: string }) => {
        csrfToken = body.csrf_token;
      })
      .catch((err) => {
        sessionPromise = null;
        throw err;
      });
  }
  return sessionPromise;
}

async function send(url: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  if (csrfToken) headers.set("X-CSRF-Token", csrfToken);
  return fetch(url, { ...init, headers, credentials: "same-origin" });
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    await ensureSession();
    res = await send(url, init);
    // The app was restarted: its old session is gone. Get a new one and retry once.
    if (res.status === 401 || (res.status === 403 && init?.method && init.method !== "GET")) {
      await ensureSession(true);
      res = await send(url, init);
    }
  } catch (err) {
    if (err instanceof Error && err.message.startsWith("Could not start")) throw err;
    throw new Error("The app's local service is not responding. Is it still running?");
  }
  if (!res.ok) {
    let detail = `Something went wrong (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep the generic message */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  gallery: () => request<{ gallery: Gallery | null }>("/api/gallery"),
  createGallery: (name: string) => request<{ id: number; name: string }>("/api/gallery", json({ name })),
  addPerson: (galleryId: number, file: File) => {
    const form = new FormData();
    form.append("file", file);
    form.append("filename", file.name);
    return request<Person & { cached: boolean }>(`/api/gallery/${galleryId}/people`, {
      method: "POST",
      body: form,
    });
  },
  finalize: (galleryId: number) =>
    request<{ gallery: Gallery }>(`/api/gallery/${galleryId}/finalize`, { method: "POST" }),
  forget: () => request<{ ok: boolean }>("/api/gallery", { method: "DELETE" }),
  identify: (file: File, threshold?: number) => {
    const form = new FormData();
    form.append("file", file);
    if (threshold !== undefined) form.append("threshold", String(threshold));
    return request<Identification>("/api/identify", { method: "POST", body: form });
  },
  rematch: (id: number, threshold: number) =>
    request<Identification>(`/api/identifications/${id}/rematch`, json({ threshold })),
  thumbUrl: (personId: number) => `/api/people/${personId}/thumb`,
};

const IMAGE_EXT = /\.(jpe?g|png|webp|bmp|tiff?|heic|heif|jfif)$/i;
export const isImageFile = (f: File) => IMAGE_EXT.test(f.name) || f.type.startsWith("image/");

/** How to describe a similarity score to a non-technical user. */
export function matchStrength(score: number): { label: string; level: "strong" | "good" | "fair" } {
  if (score >= 0.5) return { label: "Strong match", level: "strong" };
  if (score >= 0.42) return { label: "Good match", level: "good" };
  return { label: "Borderline match", level: "fair" };
}
