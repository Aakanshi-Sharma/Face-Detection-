import { Identification } from "./api";

const KNOWN = "#0f8a6c";
const UNKNOWN = "#c2740a";

function save(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const baseName = (file: string) => file.replace(/\.[^.]+$/, "") || "photo";

/** Draw boxes and names onto the photo and save it as a PNG. */
export async function downloadAnnotated(result: Identification) {
  const img = new Image();
  img.src = result.image_url;
  await img.decode();

  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0);

  const k = img.naturalWidth / result.width;
  const font = Math.max(13, Math.round(canvas.width / 75));
  const line = Math.max(2, Math.round(font / 5));
  ctx.font = `600 ${font}px "Segoe UI", system-ui, sans-serif`;
  ctx.textBaseline = "middle";

  for (const f of result.faces) {
    const [x1, y1, x2, y2] = f.box.map((v) => v * k);
    const known = f.status === "known";
    const color = known ? KNOWN : UNKNOWN;
    ctx.strokeStyle = color;
    ctx.lineWidth = line;
    ctx.setLineDash(known ? [] : [line * 3, line * 2]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);

    const text = `${f.index + 1}  ${known ? f.name : "Unknown"}`;
    const pad = font * 0.45;
    const w = ctx.measureText(text).width + pad * 2;
    const h = font * 1.55;
    const cx = (x1 + x2) / 2;
    const tx = Math.min(Math.max(cx - w / 2, 2), canvas.width - w - 2);
    const ty = y2 + line + h > canvas.height ? y1 - h - line : y2 + line;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(tx, ty, w, h, h / 4);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.fillText(text, tx + pad, ty + h / 2);
  }

  canvas.toBlob((blob) => blob && save(blob, `${baseName(result.file_name)} - named.png`), "image/png");
}

/** Save the results list as a spreadsheet-friendly CSV file. */
export function downloadCsv(result: Identification) {
  const esc = (v: string | number) => `"${String(v).replace(/"/g, '""')}"`;
  const rows = [
    ["Number", "Name", "Result", "Similarity", "Closest person", "Left", "Top", "Width", "Height"],
    ...result.faces.map((f) => [
      f.index + 1,
      f.status === "known" ? f.name ?? "" : "Unknown",
      f.status === "known" ? "In the folder" : "Not in the folder",
      f.score.toFixed(3),
      f.candidates[0]?.name ?? "",
      Math.round(f.box[0]),
      Math.round(f.box[1]),
      Math.round(f.box[2] - f.box[0]),
      Math.round(f.box[3] - f.box[1]),
    ]),
  ];
  const csv = "﻿" + rows.map((r) => r.map(esc).join(",")).join("\r\n");
  save(new Blob([csv], { type: "text/csv;charset=utf-8" }), `${baseName(result.file_name)} - names.csv`);
}
