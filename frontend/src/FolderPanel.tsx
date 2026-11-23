import { useMemo, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, FolderOpen, Search, Trash2, UserX } from "lucide-react";
import { api, Gallery } from "./api";

export interface EnrolProgress {
  total: number;
  done: number;
  current: string;
  folderName: string;
}

interface Props {
  gallery: Gallery | null;
  enrol: EnrolProgress | null;
  onPick: (files: File[]) => void;
  onForget: () => void;
}

export default function FolderPanel({ gallery, enrol, onPick, onForget }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [showIssues, setShowIssues] = useState(false);

  const choose = () => input.current?.click();
  const issues = useMemo(() => gallery?.people.filter((p) => p.status !== "ok") ?? [], [gallery]);
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (gallery?.people ?? []).filter((p) => p.status !== "failed" && (!q || p.name.toLowerCase().includes(q)));
  }, [gallery, query]);

  return (
    <aside className="panel folder-panel">
      <div className="panel-head">
        <span className="step">1</span>
        <h2>People folder</h2>
      </div>

      <input
        ref={input}
        type="file"
        multiple
        hidden
        // Lets the browser's picker choose a whole folder.
        {...({ webkitdirectory: "", directory: "" } as object)}
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = "";
          if (files.length) onPick(files);
        }}
      />

      {enrol ? (
        <div className="enrol">
          <p className="enrol-title">
            Reading <strong>{enrol.folderName}</strong>
          </p>
          <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={enrol.total} aria-valuenow={enrol.done}>
            <div style={{ width: `${(enrol.done / enrol.total) * 100}%` }} />
          </div>
          <p className="muted small">
            {enrol.done} of {enrol.total} photos · {enrol.current}
          </p>
          <p className="muted small">Learning each face. This is only needed once per folder.</p>
        </div>
      ) : !gallery ? (
        <div className="empty">
          <div className="empty-icon">
            <FolderOpen size={30} />
          </div>
          <p className="empty-title">Choose the folder of people</p>
          <p className="muted">
            One photo per person. The file name is the person's name, for example <em>Priya Sharma.jpg</em>.
          </p>
          <button className="btn primary" onClick={choose}>
            <FolderOpen size={18} /> Choose folder
          </button>
          <p className="muted small">Your browser may ask to confirm. The photos stay on this computer.</p>
        </div>
      ) : (
        <>
          <div className="folder-summary">
            <div className="folder-name" title={gallery.name}>
              <FolderOpen size={16} /> {gallery.name}
            </div>
            <div className="folder-stats">
              <span className="stat ok">
                <CheckCircle2 size={15} /> {gallery.enrolled} {gallery.enrolled === 1 ? "person" : "people"} ready
              </span>
              {issues.length > 0 && (
                <button className="stat warn link" onClick={() => setShowIssues((v) => !v)} aria-expanded={showIssues}>
                  <AlertTriangle size={15} /> {issues.length} need{issues.length === 1 ? "s" : ""} a look
                </button>
              )}
            </div>
          </div>

          {showIssues && issues.length > 0 && (
            <ul className="issues">
              {issues.map((p) => (
                <li key={p.id} className={p.status}>
                  {p.status === "failed" ? <UserX size={15} /> : <AlertTriangle size={15} />}
                  <div>
                    <strong>{p.file_name}</strong>
                    <span>{p.note}{p.status === "failed" ? " This person cannot be recognised." : ""}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}

          <label className="search">
            <Search size={15} />
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Find a person" aria-label="Find a person" />
          </label>

          <div className="people-grid">
            {visible.map((p, i) => (
              <figure key={p.id} title={p.note ?? p.name} style={{ "--i": Math.min(i, 30) } as React.CSSProperties}>
                <img src={api.thumbUrl(p.id)} alt="" loading="lazy" />
                {p.status === "warning" && <span className="flag" aria-label="Needs a look"><AlertTriangle size={11} /></span>}
                <figcaption>{p.name}</figcaption>
              </figure>
            ))}
            {visible.length === 0 && <p className="muted small">No one matches “{query}”.</p>}
          </div>

          <div className="folder-actions">
            <button className="btn" onClick={choose}>
              <FolderOpen size={16} /> Choose another folder
            </button>
            <button
              className="btn ghost danger"
              onClick={() => window.confirm("Remove these people and all results from this computer's app data?") && onForget()}
              title="Delete the stored face data"
            >
              <Trash2 size={16} /> Forget
            </button>
          </div>
        </>
      )}
    </aside>
  );
}
