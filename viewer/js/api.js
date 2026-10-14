// Talking to viewer/serve.py: static files under the project root (the page lives in viewer/, so data is ../...) and
// the POST endpoints. Long jobs answer at once ({started: true}); their progress is in runs/<run>/status.json.
export const runDir = (run) => `../runs/${run}`;
export const composeDir = (scene) => `../runs/${scene}_compose`;

export async function get(f, kind = 'arrayBuffer') {
  const r = await fetch(f, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${f}: ${r.status}`);
  return r[kind]();
}
export const getJSON = (f) => get(f, 'json');
export async function tryJSON(f) { try { return await getJSON(f); } catch (e) { return null; } }

export async function post(path, body) {
  const r = await fetch(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });
  let j = {}; try { j = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) { const e = new Error(j.error || `${path}: ${r.status}`); e.body = j; throw e; }
  return j;
}

// status.json "updated" as ms since epoch, whether the backend wrote an ISO string or seconds
export function updatedMs(st) {
  const u = st && st.updated;
  if (u === undefined || u === null) return 0;
  if (typeof u === 'number') return u > 1e12 ? u : u * 1000;
  const t = Date.parse(u); return Number.isNaN(t) ? 0 : t;
}

// <run>'s status.json "updated" right now: read it before starting a job and hand it to waitRun
export async function lastUpdate(run) { const st = await tryJSON(`${runDir(run)}/status.json`); return st ? st.updated : null; }

// wait until <run>'s job has finished: "updated" must differ from `before`, so only server times are compared (the
// browser may be on another computer, through a forwarded port, with its own clock)
export async function waitRun(run, before) {
  for (;;) {
    await new Promise((ok) => setTimeout(ok, 2000));
    const st = await tryJSON(`${runDir(run)}/status.json`);
    if (st && st.updated !== before && (st.state === 'done' || st.state === 'failed')) return st;
  }
}

const loadImg = (src) => new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => bad(new Error(src)); i.src = src; });

// the generation image with its mask on top (yellow tint, outline). zoom: show only the masked part, with a margin
export async function drawMask(canvas, cropUrl, maskUrl, px, zoom) {
  const [img, msk] = await Promise.all([loadImg(cropUrl), loadImg(maskUrl)]);
  let sx = 0, sy = 0, sw = img.width, sh = img.height;
  if (zoom) {
    const N = 256, t = document.createElement('canvas'); t.width = t.height = N;
    const tg = t.getContext('2d', { willReadFrequently: true }); tg.drawImage(msk, 0, 0, N, N); const d = tg.getImageData(0, 0, N, N).data;
    let x0 = N, y0 = N, x1 = -1, y1 = -1;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (d[(y * N + x) * 4] > 127) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    if (x1 >= 0) {
      const k = img.width / N, side = Math.max(x1 - x0, y1 - y0) * 1.3 + 4, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      sw = sh = Math.min(N, side) * k; sx = Math.max(0, Math.min(img.width - sw, (cx - side / 2) * k)); sy = Math.max(0, Math.min(img.height - sh, (cy - side / 2) * k));
    }
  }
  canvas.width = canvas.height = px;
  const g = canvas.getContext('2d', { willReadFrequently: true }); g.drawImage(img, sx, sy, sw, sh, 0, 0, px, px);
  const t = document.createElement('canvas'); t.width = t.height = px;
  const tg = t.getContext('2d', { willReadFrequently: true }); tg.drawImage(msk, sx, sy, sw, sh, 0, 0, px, px);
  const m = tg.getImageData(0, 0, px, px).data, out = g.getImageData(0, 0, px, px), d = out.data;
  const inside = (i) => m[i * 4] > 127;
  for (let y = 0; y < px; y++) for (let x = 0; x < px; x++) {
    const i = y * px + x; if (!inside(i)) continue;
    const edge = x === 0 || y === 0 || x === px - 1 || y === px - 1 || !inside(i - 1) || !inside(i + 1) || !inside(i - px) || !inside(i + px), k = i * 4;
    if (edge) { d[k] = 255; d[k + 1] = 225; d[k + 2] = 77; }
    else { d[k] = d[k] * 0.6 + 102; d[k + 1] = d[k + 1] * 0.6 + 90; d[k + 2] = d[k + 2] * 0.6 + 31; }
  }
  g.putImageData(out, 0, 0);
}
