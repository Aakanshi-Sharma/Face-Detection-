import { useEffect, useRef, useState } from "react";
import { Download, FileSpreadsheet, ImagePlus, RefreshCw, SlidersHorizontal, Upload, Users } from "lucide-react";
import { Gallery, Identification } from "./api";
import ResultsList from "./ResultsList";
import { downloadAnnotated, downloadCsv } from "./exporters";

interface Props {
  gallery: Gallery | null;
  enrolling: boolean;
  pending: { name: string; preview: string } | null;
  result: Identification | null;
  threshold: number;
  defaultThreshold: number;
  onThreshold: (value: number) => void;
  onFile: (file: File) => void;
  onClear: () => void;
}

export default function PhotoStage(props: Props) {
  const { gallery, enrolling, pending, result, threshold, defaultThreshold, onThreshold, onFile, onClear } = props;
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [active, setActive] = useState<number | null>(null);
  const [showTuning, setShowTuning] = useState(false);
  const ready = !!gallery && gallery.enrolled > 0 && !enrolling;

  useEffect(() => setActive(null), [result?.id]);

  // Paste a photo straight from the clipboard.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (file && ready && !pending) onFile(file);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [ready, pending, onFile]);

  const browse = () => input.current?.click();
  const drop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && ready) onFile(file);
  };

  const fileInput = (
    <input
      ref={input}
      type="file"
      accept="image/*,.heic,.heif"
      hidden
      onChange={(e) => {
        const file = e.target.files?.[0];
        e.target.value = "";
        if (file) onFile(file);
      }}
    />
  );

  const head = (
    <div className="panel-head">
      <span className="step">2</span>
      <h2>Group photo</h2>
    </div>
  );

  if (pending) {
    return (
      <section className="panel stage">
        {head}
        <div className="working">
          <div className="working-photo">
            <img src={pending.preview} alt="" onError={(e) => (e.currentTarget.style.visibility = "hidden")} />
            <div className="scan" />
            <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
          </div>
          <p className="working-text">
            <span className="spinner" /> Finding and naming faces in <strong>{pending.name}</strong>
            <span className="dots"><i>.</i><i>.</i><i>.</i></span>
          </p>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="panel stage">
        {head}
        {fileInput}
        <div
          className={`dropzone ${dragging ? "over" : ""} ${ready ? "" : "disabled"}`}
          onDragOver={(e) => {
            e.preventDefault();
            if (ready) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={drop}
          onClick={() => ready && browse()}
          role="button"
          tabIndex={ready ? 0 : -1}
          aria-disabled={!ready}
          onKeyDown={(e) => ready && (e.key === "Enter" || e.key === " ") && browse()}
        >
          <div className="empty-icon big">{ready ? <ImagePlus size={36} /> : <Users size={36} />}</div>
          {ready ? (
            <>
              <p className="empty-title">Drop a group photo here</p>
              <p className="muted">or click to choose one. You can also paste a copied picture.</p>
              <span className="btn primary" aria-hidden>
                <Upload size={18} /> Choose photo
              </span>
              <p className="muted small">
                Each face is compared with the {gallery!.enrolled} people in <strong>{gallery!.name}</strong>. Anyone
                else is marked “Unknown”.
              </p>
            </>
          ) : (
            <>
              <p className="empty-title">{enrolling ? "Getting the folder ready…" : "First, choose the folder of people"}</p>
              <p className="muted">
                {enrolling
                  ? "You can add the group photo as soon as the people are loaded."
                  : "Then add a group photo here to see who is in it."}
              </p>
            </>
          )}
        </div>
      </section>
    );
  }

  const { summary } = result;
  return (
    <section className="panel stage">
      {fileInput}
      <div className="result-bar">
        <div className="panel-head">
          <span className="step">2</span>
          <div>
            <h2 title={result.file_name}>{result.file_name}</h2>
            <p className="muted small">
              {summary.total === 0
                ? "No faces found"
                : `${summary.total} ${summary.total === 1 ? "face" : "faces"} · ${summary.known} named · ${summary.unknown} unknown`}
              {result.elapsed_ms ? ` · ${(result.elapsed_ms / 1000).toFixed(1)}s` : ""}
            </p>
          </div>
        </div>
        <div className="result-actions">
          <button className={`btn ghost ${showTuning ? "on" : ""}`} onClick={() => setShowTuning((v) => !v)} aria-expanded={showTuning}>
            <SlidersHorizontal size={16} /> Adjust
          </button>
          <button className="btn" onClick={() => downloadAnnotated(result)} disabled={summary.total === 0}>
            <Download size={16} /> Save picture
          </button>
          <button className="btn" onClick={() => downloadCsv(result)} disabled={summary.total === 0}>
            <FileSpreadsheet size={16} /> Save list
          </button>
          <button className="btn primary" onClick={browse}>
            <RefreshCw size={16} /> New photo
          </button>
        </div>
      </div>

      {showTuning && (
        <div className="tuning">
          <div>
            <strong>How sure must a match be?</strong>
            <p className="muted small">
              Stricter means fewer wrong names but more people marked “Unknown”. The default suits most photos.
            </p>
          </div>
          <div className="slider">
            <span className="muted small">Looser</span>
            <input
              type="range"
              min={0.2}
              max={0.6}
              step={0.01}
              value={threshold}
              onChange={(e) => onThreshold(Number(e.target.value))}
              aria-label="Match strictness"
            />
            <span className="muted small">Stricter</span>
            <output>{threshold.toFixed(2)}</output>
            <button className="btn ghost small" onClick={() => onThreshold(defaultThreshold)} disabled={Math.abs(threshold - defaultThreshold) < 0.005}>
              Reset
            </button>
          </div>
        </div>
      )}

      {summary.total === 0 ? (
        <div className="noface">
          <img src={result.image_url} alt="" />
          <div>
            <p className="empty-title">No faces found in this photo</p>
            <p className="muted">Try a sharper or closer picture where faces are clearly visible.</p>
            <button className="btn" onClick={onClear}>Choose a different photo</button>
          </div>
        </div>
      ) : (
        <div className="result-body">
          <div className="photo-wrap" onDragOver={(e) => e.preventDefault()} onDrop={drop}>
            <div className="photo" style={{ aspectRatio: `${result.width} / ${result.height}` }}>
              <img src={result.image_url} alt={`Group photo ${result.file_name}`} />
              {result.faces.map((f) => {
                const [x1, y1, x2, y2] = f.box;
                const known = f.status === "known";
                return (
                  <div
                    key={`${result.id}-${f.index}`}
                    className={`facebox ${known ? "known" : "unknown"} ${active === f.index ? "active" : ""} ${active !== null && active !== f.index ? "dim" : ""}`}
                    style={{
                      ["--i" as string]: f.index,
                      left: `${(x1 / result.width) * 100}%`,
                      top: `${(y1 / result.height) * 100}%`,
                      width: `${((x2 - x1) / result.width) * 100}%`,
                      height: `${((y2 - y1) / result.height) * 100}%`,
                    }}
                    onMouseEnter={() => setActive(f.index)}
                    onMouseLeave={() => setActive(null)}
                    onClick={() => document.getElementById(`face-row-${f.index}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" })}
                  >
                    <span className="tag">
                      <b>{f.index + 1}</b>
                      {known ? f.name : "Unknown"}
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="legend">
              <span><i className="swatch known" /> In the folder</span>
              <span><i className="swatch unknown" /> Unknown, not in the folder</span>
            </div>
          </div>
          <ResultsList result={result} active={active} onActive={setActive} />
        </div>
      )}
    </section>
  );
}
