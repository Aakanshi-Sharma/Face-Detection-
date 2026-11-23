import { useEffect, useRef, useState } from "react";
import { ArrowRight, HelpCircle } from "lucide-react";
import { api, Identification, matchStrength } from "./api";

type Filter = "all" | "known" | "unknown";

/** Animate a number towards its new value (skipped if the user prefers less motion). */
function useCountUp(target: number, ms = 500): number {
  const [value, setValue] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setValue(target);
      return;
    }
    const start = performance.now();
    const origin = from.current === target ? 0 : from.current;
    let frame = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const eased = 1 - Math.pow(1 - t, 3);
      const current = Math.round(origin + (target - origin) * eased);
      setValue(current);
      from.current = current;
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, ms]);
  return value;
}

function Count({ value }: { value: number }) {
  return <span>{useCountUp(value)}</span>;
}

interface Props {
  result: Identification;
  active: number | null;
  onActive: (index: number | null) => void;
}

export default function ResultsList({ result, active, onActive }: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const { summary } = result;
  const faces = result.faces.filter((f) => filter === "all" || f.status === filter);

  const tabs: { key: Filter; label: string; count: number }[] = [
    { key: "all", label: "Everyone", count: summary.total },
    { key: "known", label: "Named", count: summary.known },
    { key: "unknown", label: "Unknown", count: summary.unknown },
  ];

  return (
    <div className="results">
      <div className="tabs" role="tablist">
        {tabs.map((t) => (
          <button key={t.key} role="tab" aria-selected={filter === t.key} className={filter === t.key ? "on" : ""} onClick={() => setFilter(t.key)}>
            {t.label} <Count value={t.count} />
          </button>
        ))}
      </div>

      <ol className="rows">
        {faces.map((f) => {
          const known = f.status === "known";
          const strength = known ? matchStrength(f.score) : null;
          return (
            <li
              key={`${result.id}-${f.index}`}
              id={`face-row-${f.index}`}
              style={{ "--i": f.index } as React.CSSProperties}
              className={`row ${known ? "known" : "unknown"} ${active === f.index ? "active" : ""}`}
              onMouseEnter={() => onActive(f.index)}
              onMouseLeave={() => onActive(null)}
            >
              <span className="num">{f.index + 1}</span>
              <div className="pair">
                <img src={f.crop} alt="Face from the group photo" />
                {known && f.person_id !== null ? (
                  <>
                    <ArrowRight size={14} className="arrow" />
                    <img src={api.thumbUrl(f.person_id)} alt="Photo from the folder" />
                  </>
                ) : (
                  <>
                    <ArrowRight size={14} className="arrow" />
                    <span className="nomatch"><HelpCircle size={18} /></span>
                  </>
                )}
              </div>
              <div className="who">
                <strong>{known ? f.name : "Unknown"}</strong>
                {known && strength ? (
                  <span className={`strength ${strength.level}`}>
                    <i style={{ width: `${Math.min(100, Math.max(8, ((f.score - 0.2) / 0.5) * 100))}%` }} />
                    <em>{strength.label} · {f.score.toFixed(2)}</em>
                  </span>
                ) : (
                  <span className="muted small">
                    {f.hint ? `Not sure. Closest is ${f.hint.name} (${f.hint.score.toFixed(2)})` : "Not in the folder"}
                  </span>
                )}
              </div>
            </li>
          );
        })}
        {faces.length === 0 && <li className="muted small none">Nobody in this group.</li>}
      </ol>
    </div>
  );
}
