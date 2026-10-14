// One object's check (what debug.html used to be): the 3DGS before / after deletion, the deleted points over it, the
// closure box and the cameras, per-view 2D comparisons, the point brush, the mesh eraser, an in-browser preview of the
// result, and saving the hand edits (the server then redoes the deletion in the background).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { renderer, scene, camera, world, controls, toThree, size, local, onFrame, resized, makeSplat, SplatFormat, keepCamera, belowBar } from './stage.js';
import { runDir, get, post, waitRun } from './api.js';

export async function startCheck(run, side) {
  const base = `${runDir(run)}/debug`;
  const statusEl = document.getElementById('status');
  const obj = side.objects().find((o) => o.run === run);
  let note = '';

  const meta = await get(`${base}/meta.json`, 'json');
  const title = obj ? (obj.label || obj.name) : meta.name;
  const n = meta.n;
  const pbuf = await get(`${base}/points.bin`);
  const pos = new Float32Array(pbuf, 0, n * 3), att = new Uint8Array(pbuf, n * 12, n * 4);
  const index = new Uint32Array(await get(`${base}/index.bin`));
  const beforeBytes = new Uint8Array(await get(`${base}/before.splat`));
  const beforeIndex = new Uint32Array(await get(`${base}/before_index.bin`));
  const solid = new Uint8Array(await get(`${base}/mesh_vox.bin`));
  let maxIdx = 0; for (const a of [index, beforeIndex]) for (let k = 0; k < a.length; k++) if (a[k] > maxIdx) maxIdx = a[k];
  const where = new Map(); for (let i = 0; i < n; i++) where.set(index[i], i);
  const manual = new Map();                                     // point i -> 'del' | 'keep'
  try {
    const m = await get(`${runDir(run)}/manual.json`, 'json');
    for (const s of m.delete || []) if (where.has(s)) manual.set(where.get(s), 'del');
    for (const s of m.keep || []) if (where.has(s)) manual.set(where.get(s), 'keep');
  } catch (e) { /* no hand edits yet */ }
  const floorY = meta.closure.floor_y, floorLayer = 0.03 * meta.unit;
  const height = new Float32Array(n); for (let i = 0; i < n; i++) height[i] = floorY - pos[i * 3 + 1];
  const pick = new Uint8Array(n);                               // what the mesh eraser covers right now
  let eraserOn = false;
  const center = toThree(meta.center);

  // ---- 3DGS layer: before / after. "after" starts as the server's file and is rebuilt here by "预览删点后"
  const splats = { before: makeSplat(`${base}/before.splat`).v, after: makeSplat(`${base}/after.splat`).v };
  scene.add(splats.before);
  function setScene(m) {
    for (const k of ['before', 'after']) scene.remove(splats[k]);
    if (m !== 'none') scene.add(splats[m]);
    document.querySelectorAll('[data-scene]').forEach((b) => b.classList.toggle('on', b.dataset.scene === m));
  }
  document.querySelectorAll('[data-scene]').forEach((b) => b.onclick = () => setScene(b.dataset.scene));

  // ---- point overlay: everything that is (or would be) deleted, drawn over the scene
  const willDelete = (i) => { const m = manual.get(i); if (m) return m === 'del'; return att[i * 4 + 3] === 0 || (eraserOn && pick[i] === 1); };
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const col = new Float32Array(n * 3); geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const pmat = new THREE.PointsMaterial({ size: 2.5, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.8, depthTest: false });
  const points = new THREE.Points(geom, pmat); points.frustumCulled = false; points.renderOrder = 10; world.add(points);
  let unsaved = false;
  function refresh() {
    const idx = []; let auto = 0, hd = 0, hk = 0, er = 0;
    for (let i = 0; i < n; i++) {
      const m = manual.get(i); let c = null;
      if (m === 'del') { c = [1, .31, .85]; hd++; }
      else if (m === 'keep') { c = [.2, .84, 1]; hk++; }
      else if (att[i * 4 + 3] === 0) { c = [1, .16, .16]; auto++; }
      else if (eraserOn && pick[i]) { c = [.69, .36, 1]; er++; }
      if (c) { idx.push(i); col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2]; }
    }
    geom.setIndex(new THREE.BufferAttribute(Uint32Array.from(idx), 1)); geom.attributes.color.needsUpdate = true;
    document.getElementById('eCount').textContent = eraserOn ? `橡皮擦会多删 ${er} 个点（紫）` : '';
    statusEl.textContent = `${title}：自动删 ${auto}` + (hd || hk ? `，手动删 ${hd} / 手动留 ${hk}` : '') + (er ? `，橡皮擦会删 ${er}` : '') +
      (unsaved || er ? '（有修改还没保存）' : '') + (note ? `\n${note}` : '') + '\n左键拖：转  滚轮：远近  右键拖：平移';
  }
  const tglBtn = (id, getv, setv) => { const b = document.getElementById(id); b.onclick = () => { setv(!getv()); b.classList.toggle('on', getv()); }; };
  tglBtn('tPts', () => points.visible, (x) => { points.visible = x; });

  // ---- new mesh, deletion closure, cameras
  let mesh = null;
  new GLTFLoader().load(`${base}/${meta.mesh}`, (g) => {
    g.scene.traverse((m) => { if (m.isMesh) m.material = new THREE.MeshBasicMaterial({ map: m.material.map || null, side: THREE.DoubleSide }); });
    mesh = g.scene; mesh.visible = false; world.add(mesh);
  });
  tglBtn('tMesh', () => !!(mesh && mesh.visible), (x) => { if (mesh) mesh.visible = x; });
  const box = new THREE.Group(); world.add(box);
  { const c = meta.closure, y0 = c.floor_y - c.bottom, y1 = c.floor_y - c.top, P = c.poly, pts = [];
    for (let i = 0; i < P.length; i++) {
      const a = P[i], b = P[(i + 1) % P.length];
      pts.push(new THREE.Vector3(a[0], y0, a[1]), new THREE.Vector3(b[0], y0, b[1]), new THREE.Vector3(a[0], y1, a[1]), new THREE.Vector3(b[0], y1, b[1]));
      if (i % Math.max(1, Math.floor(P.length / 12)) === 0) pts.push(new THREE.Vector3(a[0], y0, a[1]), new THREE.Vector3(a[0], y1, a[1]));
    }
    const ln = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xffe14d, depthTest: false }));
    ln.renderOrder = 12; box.add(ln); }
  tglBtn('tBox', () => box.visible, (x) => { box.visible = x; });
  const cams = new THREE.Group(); world.add(cams);
  for (const v of meta.views) {
    const C = new THREE.Vector3(...v.C), Rm = v.Rb, d = 0.07, hh = v.size / 2 / v.f * d;
    const ax = (j) => new THREE.Vector3(Rm[0][j], Rm[1][j], Rm[2][j]);
    const corner = (sx, sy) => C.clone().add(ax(0).multiplyScalar(sx * hh)).add(ax(1).multiplyScalar(sy * hh)).add(ax(2).multiplyScalar(d));
    const cs = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)], pts = [];
    for (let j = 0; j < 4; j++) pts.push(C, cs[j], cs[j], cs[(j + 1) % 4]);
    const ln = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: !v.mask ? 0x777777 : v.weight > 1 ? 0x3ddc84 : 0x4ea3ff, depthTest: false }));
    ln.renderOrder = 12; cams.add(ln);
  }
  tglBtn('tCams', () => cams.visible, (x) => { cams.visible = x; });

  // ---- mesh eraser: a violet copy of the placed mesh, always drawn on top (the splats hid it from some angles);
  //      scaled about its centre / moved by the sliders; the points it covers are worked out here, live,
  //      against the mesh's solid voxels (debug/mesh_vox.bin)
  let ghost = null;
  new GLTFLoader().load(`${base}/${meta.mesh}`, (g) => {
    const grp = new THREE.Group(); g.scene.updateMatrixWorld(true);
    g.scene.traverse((m) => {
      if (!m.isMesh) return;
      const fill = new THREE.Mesh(m.geometry, new THREE.MeshBasicMaterial({ color: 0xb05cff, transparent: true, opacity: 0.22, depthTest: false, depthWrite: false, side: THREE.DoubleSide }));
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry, 35), new THREE.LineBasicMaterial({ color: 0xd9a6ff, depthTest: false }));
      for (const o of [fill, edges]) { o.applyMatrix4(m.matrixWorld); o.renderOrder = 20; o.frustumCulled = false; grp.add(o); }
    });
    ghost = grp; ghost.visible = eraserOn; world.add(ghost); applyEraser();
  });
  const ev = (id) => +document.getElementById(id).value;
  function eraserParams() { const s = ev('eS'); return { S: [s * ev('eX'), s * ev('eY'), s * ev('eZ')], D: [ev('eDX'), -ev('eDY'), ev('eDZ')] }; }  // world y is down
  const vx = meta.vox, [nx, ny, nz] = vx.shape, c0 = meta.mesh_center;
  let pending = false;
  function applyEraser() {
    for (const id of ['eS', 'eX', 'eY', 'eZ', 'eDX', 'eDY', 'eDZ']) document.getElementById(id + 'v').textContent = ev(id).toFixed(id.startsWith('eD') ? 3 : 2);
    const { S, D } = eraserParams();
    if (ghost) { ghost.scale.set(S[0], S[1], S[2]); ghost.position.set(c0[0] - S[0] * c0[0] + D[0], c0[1] - S[1] * c0[1] + D[1], c0[2] - S[2] * c0[2] + D[2]); }
    if (!eraserOn) return;
    pick.fill(0);
    for (let i = 0; i < n; i++) {
      if (height[i] <= floorLayer) continue;
      const lx = c0[0] + (pos[i * 3] - c0[0] - D[0]) / S[0], ly = c0[1] + (pos[i * 3 + 1] - c0[1] - D[1]) / S[1], lz = c0[2] + (pos[i * 3 + 2] - c0[2] - D[2]) / S[2];
      const a = Math.round((lx - vx.origin[0]) / vx.pitch), b = Math.round((ly - vx.origin[1]) / vx.pitch), c = Math.round((lz - vx.origin[2]) / vx.pitch);
      if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz) continue;
      if (solid[(a * ny + b) * nz + c]) pick[i] = 1;
    }
    refresh();
  }
  for (const id of ['eS', 'eX', 'eY', 'eZ', 'eDX', 'eDY', 'eDZ'])
    document.getElementById(id).oninput = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; applyEraser(); }); } };
  const tErase = document.getElementById('tErase'), ePanel = document.getElementById('eraser');
  const placePanel = () => { ePanel.style.top = `${belowBar()}px`; };
  resized(placePanel);
  tErase.onclick = () => {
    eraserOn = !eraserOn; ePanel.style.display = eraserOn ? 'block' : 'none'; placePanel(); if (ghost) ghost.visible = eraserOn;
    tErase.classList.toggle('on', eraserOn); tErase.textContent = eraserOn ? '关闭' : '打开';
    if (eraserOn) applyEraser(); else { pick.fill(0); refresh(); }
  };
  document.getElementById('eReset').onclick = () => {
    for (const [id, v] of [['eS', 1.05], ['eX', 1], ['eY', 1], ['eZ', 1], ['eDX', 0], ['eDY', 0], ['eDZ', 0]]) document.getElementById(id).value = v;
    applyEraser();
  };

  // ---- preview: rebuild "after" from before.splat here, dropping what is (or would be) deleted -- no server round trip
  document.getElementById('preview').onclick = async () => {
    const t0 = performance.now();
    const del = new Uint8Array(maxIdx + 1);
    for (let i = 0; i < n; i++) if (willDelete(i)) del[index[i]] = 1;
    let kept = 0; for (let k = 0; k < beforeIndex.length; k++) if (!del[beforeIndex[k]]) kept++;
    const out = new Uint8Array(kept * 32);
    for (let k = 0, j = 0; k < beforeIndex.length; k++) if (!del[beforeIndex[k]]) { out.set(beforeBytes.subarray(k * 32, k * 32 + 32), j * 32); j++; }
    note = '预览生成中…'; refresh();
    const { v, ready } = makeSplat(URL.createObjectURL(new Blob([out])), SplatFormat.Splat);
    await ready;
    const old = splats.after; scene.remove(old); splats.after = v; setScene('after');
    try { old.dispose && old.dispose(); } catch (e) { /* ignore */ }
    note = `预览：删掉 ${beforeIndex.length - kept} 个 3DGS 点，用时 ${((performance.now() - t0) / 1000).toFixed(1)} 秒（还没保存的修改也算进去了）`; refresh();
  };

  // ---- save: hand edits + what the eraser covers -> manual.json; the server redoes the deletion, the comparison
  //      images, this page's data and the composed scene in the background (the object list shows the progress)
  async function postEdits() {
    const d = [], k = [];
    for (const [i, m] of manual) (m === 'del' ? d : k).push(index[i]);
    await post(`/api/manual?run=${run}`, { delete: d, keep: k });
    const t0 = Date.now();
    note = '已保存，后台更新中（删点、对比图、合成场景）…'; refresh();
    const j = await post(`/api/apply?run=${run}`);
    if (j.started) {
      const st = await waitRun(run, t0);
      note = st.state === 'done' ? '已保存，后台也更新完了' : `后台更新失败：${st.error || ''}`;
    } else note = '已保存，后台也更新完了';
    refresh();
  }
  document.getElementById('save').onclick = async () => {
    if (eraserOn) for (let i = 0; i < n; i++) if (pick[i] && !manual.has(i) && att[i * 4 + 3] !== 0) manual.set(i, 'del');
    pick.fill(0); unsaved = false; refresh();
    try { await postEdits(); } catch (e) { note = '保存失败：' + e.message; refresh(); }
  };
  document.getElementById('undo').onclick = async () => {
    if (!confirm('清空这个物体所有手动修改（点刷子和网格擦除），回到自动结果？')) return;
    manual.clear(); pick.fill(0); unsaved = false; refresh();
    try { await postEdits(); } catch (e) { note = '清空失败：' + e.message; refresh(); }
  };

  // ---- point brush: marks the front-most points under the circle (all points near the object, kept or removed)
  const brushEl = document.getElementById('brush'); let brushMode = 'off', painting = false, pend = null; const R = 18;
  document.querySelectorAll('[data-brush]').forEach((b) => b.onclick = () => {
    brushMode = b.dataset.brush; controls.enabled = brushMode === 'off'; brushEl.style.display = brushMode === 'off' ? 'none' : 'block';
    document.querySelectorAll('[data-brush]').forEach((x) => x.classList.toggle('on', x === b));
  });
  const M4 = new THREE.Matrix4();
  function paint(cx, cy) {
    M4.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(world.matrixWorld);
    const e = M4.elements, { w: W, h: H } = size(), hit = []; let wmin = Infinity;
    for (let i = 0; i < n; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      const w = e[3] * x + e[7] * y + e[11] * z + e[15]; if (w <= 0.01) continue;
      const sx = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / w + 1) / 2 * W, sy = (1 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / w) / 2 * H;
      if ((sx - cx) ** 2 + (sy - cy) ** 2 <= R * R) { hit.push([i, w]); if (w < wmin) wmin = w; }
    }
    for (const [i, w] of hit) if (w < wmin + 0.03) manual.set(i, brushMode);       // front layer only
    unsaved = true;
  }
  renderer.domElement.addEventListener('pointerdown', (e) => { if (brushMode !== 'off') { painting = true; pend = local(e); } });
  addEventListener('pointerup', () => { if (painting) { painting = false; refresh(); } });
  renderer.domElement.addEventListener('pointermove', (e) => {
    brushEl.style.left = `${e.clientX - R}px`; brushEl.style.top = `${e.clientY - R}px`; brushEl.style.width = brushEl.style.height = `${2 * R}px`;
    if (painting) pend = local(e);
  });
  onFrame(() => { if (painting && pend) { paint(pend[0], pend[1]); pend = null; refresh(); } });

  // ---- views: jump the 3D camera there, and show that view's 2D comparison (mask outline in yellow)
  const sel = document.getElementById('views'), panel = document.getElementById('panel'), pimg = document.getElementById('pimg');
  let cur = null, imgKind = 'before';
  sel.add(new Option('自由视角', ''));
  for (const v of meta.views) if (v.mask) sel.add(new Option(`${v.k}. ${v.name}${v.weight > 1 ? '（照片/主视角）' : ''}`, v.k));
  function showImg() {
    if (cur === null) { panel.style.display = 'none'; return; }
    panel.style.display = 'block'; pimg.src = `${runDir(run)}/views/${cur.k}/cmp_${imgKind}.jpg?t=${Date.now()}`;
    document.getElementById('pcap').textContent = { before: '原来的 3DGS，黄线 = 这个视角的抠图', after: '删点后（已保存的结果），黄线里紫色 = 该删没删掉的', removed: '只画删掉的点，黄线 = 抠图' }[imgKind];
  }
  function flyTo(v) {
    camera.fov = 2 * Math.atan(v.size / 2 / v.f) * 180 / Math.PI; camera.updateProjectionMatrix();
    camera.position.copy(toThree(v.C)); controls.target.copy(center); controls.update();
  }
  sel.onchange = () => { const v = meta.views.find((x) => String(x.k) === sel.value); cur = v || null; if (v) flyTo(v); showImg(); };
  document.querySelectorAll('[data-img]').forEach((b) => b.onclick = () => {
    imgKind = b.dataset.img; document.querySelectorAll('[data-img]').forEach((x) => x.classList.toggle('on', x === b)); showImg();
  });

  // ---- start where you were (same object), else from the photo / generation view
  const start = meta.views.find((v) => v.weight > 1 && v.mask) || meta.views[0];
  if (!keepCamera(`cam:${run}`)) flyTo(start);
  sel.value = String(start.k); cur = start; showImg();
  refresh();
}
