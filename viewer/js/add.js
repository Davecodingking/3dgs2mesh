// Adding an object: frame it in the yellow square (the square is exactly the image the server will make), type its
// English name and click on it. The server renders that view from the 3DGS -- or, when standing at the panorama camera,
// cuts it from the panorama photo -- and runs SAM3 at the click. The mask comes back for a look; "开始生成" then runs
// the rest of the pipeline in the background and the object list shows the progress.
import * as THREE from 'three';
import { renderer, camera, controls, el, local, size, resized, belowBar } from './stage.js';
import { runDir, post, drawMask } from './api.js';

const S = 1024;                                                 // the generation image is S x S pixels
const $ = (id) => document.getElementById(id);

export function setupAdd({ sceneId, side, setGS, showMeshes }) {
  const frame = $('frame'), mark = $('clickMark'), panel = $('addPanel'), msgEl = $('aMsg'), img = $('aImg'), goBtn = $('aGo');
  let on = false, run = null, mine = false, busy = false, pano = false;
  const say = (t, bad) => { msgEl.textContent = t; msgEl.className = bad ? 'err' : 'hint'; };

  // the square: centred on the canvas, because that is where the three.js camera's optical axis goes through
  function square() { const r = el.getBoundingClientRect(), s = Math.round(Math.min(r.width, r.height) * 0.84); return { s, x0: (r.width - s) / 2, y0: (r.height - s) / 2, r }; }
  function layout() {
    if (!on) return;
    const { s, x0, y0, r } = square();
    Object.assign(frame.style, { left: `${r.left + x0}px`, top: `${r.top + y0}px`, width: `${s}px`, height: `${s}px` });
    panel.style.top = `${belowBar()}px`;
  }
  resized(layout);

  // the browser camera as the server wants it: OpenCV camera axes (x right, y down, z forward) as the columns of Rb, in
  // the OpenCV world, and the focal length in pixels of the S x S image the square covers. A three.js camera looks down
  // -z with y up, and the world is the OpenCV one turned 180 deg about x, so Rb = F R diag(1,-1,-1) with F = diag(1,-1,-1)
  function serverCam() {
    const { s } = square(), { h } = size();
    camera.updateMatrixWorld();
    const m = camera.matrixWorld.elements;                      // column major: x axis m[0..2], y axis m[4..6], z axis m[8..10]
    const X = [m[0], -m[1], -m[2]], Y = [-m[4], m[5], m[6]], Z = [-m[8], m[9], m[10]];
    const fpx = (h / 2) / Math.tan(camera.fov * Math.PI / 360);
    const p = camera.position;
    return { C: pano ? [0, 0, 0] : [p.x, -p.y, -p.z], Rb: [0, 1, 2].map((r) => [X[r], Y[r], Z[r]]), f: fpx * S / s, size: S };
  }
  // and back: put the browser camera where an object was framed
  function flyToCam(cam) {
    const { s } = square(), { h } = size();
    const atPano = cam.C.every((v) => v === 0);
    camera.fov = 2 * Math.atan((h / 2) / (cam.f * s / S)) * 180 / Math.PI; camera.updateProjectionMatrix();
    camera.position.set(cam.C[0], -cam.C[1], -cam.C[2]);
    const fwd = new THREE.Vector3(cam.Rb[0][2], -cam.Rb[1][2], -cam.Rb[2][2]);
    controls.target.copy(camera.position).add(fwd.multiplyScalar(atPano ? 0.05 : 0.6)); controls.update();
    if (atPano !== pano) setPano(atPano);
  }

  // standing at the panorama camera: every camera move is shifted back to the origin (the view direction stays), so
  // dragging turns the head; the wheel zooms by field of view. The server then cuts the image from the photo.
  controls.addEventListener('change', () => {
    if (!pano) return;
    const p = camera.position; if (p.lengthSq() < 1e-12) return;
    controls.target.sub(p); p.set(0, 0, 0);
  });
  renderer.domElement.addEventListener('wheel', (e) => {
    if (!pano) return;
    e.preventDefault(); camera.fov = Math.min(100, Math.max(25, camera.fov * Math.exp(e.deltaY * 0.001))); camera.updateProjectionMatrix();
  }, { passive: false });
  function setPano(x) {
    pano = x; controls.enablePan = !x; controls.enableZoom = !x; controls.rotateSpeed = x ? -0.35 : 1;
    const d = new THREE.Vector3().subVectors(controls.target, camera.position).normalize();
    if (x) { camera.position.set(0, 0, 0); controls.target.copy(d.multiplyScalar(0.05)); }
    else controls.target.copy(camera.position).add(d.multiplyScalar(0.6));
    controls.update();
    $('aPano').classList.toggle('on', x); $('aPano').textContent = x ? '在全景相机位置（点这里自由移动）' : '站到全景相机位置';
    $('aPanoNote').textContent = x ? '用全景照片出图，最清楚。拖动是转头，滚轮是缩放。' : '';
  }
  $('aPano').onclick = () => setPano(!pano);

  function uniqueName(prompt) {                                 // ascii, used in file names; unique in the scene
    const w = prompt.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').trim().split(/\s+/).pop() || 'object';
    const used = new Set(side.objects().map((o) => o.name));
    let name = w, k = 2; while (used.has(name)) name = `${w}${k++}`;
    return name;
  }
  async function showMask() {
    const t = Date.now();
    await drawMask(img, `${runDir(run)}/crop.jpg?t=${t}`, `${runDir(run)}/mask_final.png?t=${t}`, 640, false);
    img.style.display = 'block'; goBtn.style.display = '';
  }

  // a click (not a drag) inside the square: make the image there and cut the object out
  async function probe(e) {
    if (busy) return;
    const [px, py] = local(e), { s, x0, y0 } = square();
    if (px < x0 || py < y0 || px >= x0 + s || py >= y0 + s) return say('请点在黄框里面。', true);
    const prompt = $('aPrompt').value.trim();
    if (!/^[A-Za-z][A-Za-z0-9 -]*$/.test(prompt)) return say('先在上面填英文词（SAM3 只认英文），再点物体。', true);
    Object.assign(mark.style, { display: 'block', left: `${e.clientX}px`, top: `${e.clientY}px` });
    const fields = {
      prompt, label: $('aLabel').value.trim(), parts: $('aParts').value.split(/[,，]/).map((t) => t.trim()).filter(Boolean),
      rest: document.querySelector('input[name=aRest]:checked').value, cover: $('aCover').checked,
      cam: serverCam(), click: [Math.round((px - x0) * S / s), Math.round((py - y0) * S / s)],
    };
    if (!run) fields.name = uniqueName(prompt);
    busy = true; goBtn.style.display = 'none'; img.style.display = 'none';
    say(`${pano ? '从全景照片裁图' : '从 3DGS 渲染'}，再用 SAM3 抠图，要几十秒…`);
    try {
      const j = await post(`/api/object?scene=${sceneId}` + (run ? `&run=${run}` : ''), fields);
      if (!run) mine = true;
      run = j.run;
      await showMask();
      say('黄色是抠出来的物体。对的话点"开始生成"；不对就改词或换个位置，在框里重新点。');
    } catch (err) {
      if (err.body && err.body.run && !run) { run = err.body.run; mine = true; }
      say('没成功：' + err.message, true);
    }
    busy = false; side.poll();
  }
  // a click is a press and release without moving (no time limit: a slow frame between the two must not turn it into a drag)
  let down = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { if (on && e.button === 0) down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!on || !down) return;
    const [x, y] = down; down = null;
    if (Math.hypot(e.clientX - x, e.clientY - y) < 6) probe(e);
  });

  goBtn.onclick = async () => {
    try { await post(`/api/run?run=${run}&from=gen`); } catch (e) { return say('开跑失败：' + e.message, true); }
    run = null; mine = false; exit(); side.poll();
  };
  $('aCancel').onclick = async () => {                          // an object made here and not started goes away again
    if (run && mine) { try { await post(`/api/remove?run=${run}`); } catch (e) { /* the list still shows it */ } }
    run = null; mine = false; exit(); side.poll();
  };

  function enter(existing) {
    if (on) return;
    on = true; setGS('orig'); showMeshes(false);
    frame.style.display = 'block'; panel.style.display = 'block'; layout();
    run = null; mine = false; img.style.display = 'none'; goBtn.style.display = 'none'; mark.style.display = 'none';
    const o = existing && existing !== 'new' ? side.objects().find((x) => x.run === existing) : null;
    if (!o) { say('1. 转动视角，把物体放进黄框，占框的一半左右最好。\n2. 填英文词，在框里点一下物体。'); return; }
    run = o.run;
    $('aPrompt').value = o.prompt || ''; $('aLabel').value = o.label || ''; $('aParts').value = (o.parts || []).join(', ');
    document.querySelector(`input[name=aRest][value=${o.rest === 'table' ? 'table' : 'floor'}]`).checked = true; $('aCover').checked = !!o.cover;
    if (o.cam && o.cam.Rb) flyToCam(o.cam);
    showMask().then(() => say('这是之前抠的图。对的话点"开始生成"；不对就在框里重新点。'))
      .catch(() => say('还没有抠图结果，在框里点一下物体。'));
  }
  function exit() {
    on = false; if (pano) setPano(false);
    frame.style.display = panel.style.display = mark.style.display = 'none';
    setGS('after'); showMeshes(true);
    const q = new URLSearchParams(location.search);
    if (q.has('add')) { q.delete('add'); history.replaceState(null, '', `?${q}`); }
  }
  return { enter, serverCam, square, setPano };
}
