import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, ShieldCheck, WifiOff, X } from "lucide-react";
import { api, Gallery, Health, Identification, isImageFile } from "./api";
import FolderPanel, { EnrolProgress } from "./FolderPanel";
import PhotoStage from "./PhotoStage";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [gallery, setGallery] = useState<Gallery | null>(null);
  const [enrol, setEnrol] = useState<EnrolProgress | null>(null);
  const [result, setResult] = useState<Identification | null>(null);
  const [pending, setPending] = useState<{ name: string; preview: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const defaultThreshold = health?.threshold ?? 0.35;
  const [threshold, setThreshold] = useState<number | null>(null);
  const rematchTimer = useRef<number>();

  // Wait for the local service and its models, then restore the last folder.
  useEffect(() => {
    let stop = false;
    let timer: number;
    const poll = async () => {
      try {
        const h = await api.health();
        if (stop) return;
        setHealth(h);
        if (!h.ready) timer = window.setTimeout(poll, 1200);
      } catch {
        if (!stop) timer = window.setTimeout(poll, 2000);
      }
    };
    poll();
    api.gallery().then((g) => !stop && setGallery(g.gallery)).catch(() => undefined);
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, []);

  const openFolder = useCallback(async (picked: File[]) => {
    const files = picked.filter(isImageFile).sort((a, b) => a.name.localeCompare(b.name));
    if (!files.length) {
      setError("That folder has no photos in it. Choose a folder with one photo per person.");
      return;
    }
    const rel = (files[0] as File & { webkitRelativePath?: string }).webkitRelativePath || "";
    const folderName = rel.split("/")[0] || "People";
    setError(null);
    setResult(null);
    setPending(null);
    setGallery(null);
    setEnrol({ total: files.length, done: 0, current: files[0].name, folderName });
    try {
      const { id } = await api.createGallery(folderName);
      let done = 0;
      let next = 0;
      // Two at a time keeps the laptop responsive while staying quick.
      const worker = async () => {
        while (next < files.length) {
          const file = files[next++];
          try {
            await api.addPerson(id, file);
          } catch (e) {
            if (String(e).includes("not responding") || String(e).includes("Model files")) throw e;
          }
          done += 1;
          setEnrol({ total: files.length, done, current: file.name, folderName });
        }
      };
      await Promise.all([worker(), worker()]);
      const { gallery: g } = await api.finalize(id);
      setGallery(g);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setEnrol(null);
    }
  }, []);

  const identify = useCallback(
    async (file: File) => {
      if (!isImageFile(file)) {
        setError("That file is not a photo. Use a JPG, PNG, WEBP or HEIC picture.");
        return;
      }
      setError(null);
      setResult(null);
      setThreshold(null);
      const preview = URL.createObjectURL(file);
      setPending({ name: file.name, preview });
      try {
        setResult(await api.identify(file));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setPending(null);
        URL.revokeObjectURL(preview);
      }
    },
    [],
  );

  const changeThreshold = useCallback(
    (value: number) => {
      setThreshold(value);
      window.clearTimeout(rematchTimer.current);
      if (!result) return;
      rematchTimer.current = window.setTimeout(async () => {
        try {
          setResult(await api.rematch(result.id, value));
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }, 120);
    },
    [result],
  );

  const forget = useCallback(async () => {
    try {
      await api.forget();
      setGallery(null);
      setResult(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const status = !health
    ? { cls: "wait", text: "Starting…" }
    : !health.models_present
      ? { cls: "bad", text: "Model files missing" }
      : health.ready
        ? { cls: "ok", text: "Ready" }
        : { cls: "wait", text: "Loading face models…" };

  return (
    <div className="app">
      <div className="backdrop" aria-hidden>
        <span /><span /><span />
      </div>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <svg viewBox="0 0 32 32" width="28" height="28">
              <rect width="32" height="32" rx="8" fill="var(--accent)" />
              <circle cx="16" cy="13" r="5" fill="var(--on-accent)" />
              <path d="M6.5 26c1.200-4.600 5-7 9.500-7s8.300 2.400 9.500 7" fill="none" stroke="var(--on-accent)" strokeWidth="2.600" strokeLinecap="round" />
            </svg>
          </span>
          <div>
            <h1>Who is in the photo?</h1>
            <p>Name everyone in a group photo from your folder of people.</p>
          </div>
        </div>
        <div className="top-chips">
          <span className="chip" title="Photos never leave this computer. No internet is used.">
            <WifiOff size={14} /> Works offline
          </span>
          <span className={`chip status ${status.cls}`}>
            {status.cls === "ok" ? <ShieldCheck size={14} /> : <span className="dot" />} {status.text}
          </span>
        </div>
      </header>

      {health && !health.models_present && (
        <div className="banner bad" role="alert">
          <AlertCircle size={18} />
          <span>
            The face models are not installed yet. Run <code>python scripts/download_models.py</code> once, then
            restart the app.
          </span>
        </div>
      )}
      {error && (
        <div className="banner bad" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
          <button className="icon-btn" onClick={() => setError(null)} aria-label="Dismiss">
            <X size={16} />
          </button>
        </div>
      )}

      <main className="layout">
        <FolderPanel gallery={gallery} enrol={enrol} onPick={openFolder} onForget={forget} />
        <PhotoStage
          gallery={gallery}
          enrolling={enrol !== null}
          pending={pending}
          result={result}
          threshold={threshold ?? result?.threshold ?? defaultThreshold}
          defaultThreshold={defaultThreshold}
          onThreshold={changeThreshold}
          onFile={identify}
          onClear={() => setResult(null)}
        />
      </main>
    </div>
  );
}
