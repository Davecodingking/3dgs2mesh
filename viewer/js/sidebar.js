// The object list on the left: the scene picker (scenes/index.json), a row for the whole scene, and one row per object
// in the scene's list (scenes/<id>.json) with its progress (runs/<run>/status.json, read every 2 s), its scores, and the
// actions that make sense right now. Rows are kept per run and only patched, so a poll neither closes an open menu nor
// redraws the thumbnails.
import { runDir, composeDir, tryJSON, post, drawMask, updatedMs } from './api.js';

export const STEP = { view: '出图', align: '对齐图', mask: '抠图', gen: '生成网格', preview: '预览图', place: '摆放', views: '多视角渲染',
  vmask: '多视角抠图', vote: '投票', carve: '删点', fit: '多视角微调', overlay: '轮廓图', check: '检查图', compare: '对比图', debug: '导出检查数据' };

// what a row says and offers, from its last job: a full run and an apply both end with "debug"; adding an object stops
// after "mask" for the user to look at the mask
export function phase(st) {
  if (!st) return { key: 'none', text: '还没跑', cls: '' };
  const step = STEP[st.step] || st.step || '';
  if (st.state === 'queued') return { key: 'busy', text: '排队等显卡…', cls: 'wait', frac: 0 };
  if (st.state === 'running') return { key: 'busy', text: `${step}（第 ${Math.min(st.n, st.i + 1)}/${st.n} 步）`, cls: 'wait', frac: st.n ? st.i / st.n : 0 };  // i = steps finished
  if (st.state === 'failed') return { key: 'failed', text: st.error === '已停止' ? '已停止' : `失败（${step}）`, cls: 'bad' };
  if (st.step === 'mask') return { key: 'probe', text: '抠图好了，等你确认', cls: 'wait' };
  if (st.step === 'debug') return { key: 'done', text: '完成', cls: 'ok' };
  return { key: 'stopped', text: `停在「${step}」`, cls: '' };
}

export async function startSidebar(sceneId, scenes, currentRun) {
  const sel = document.getElementById('sceneSel'), rowsEl = document.getElementById('rows');
  for (const s of (scenes.length ? scenes : [{ id: sceneId }])) sel.add(new Option(s.id, s.id, false, s.id === sceneId));
  sel.onchange = () => { location.search = `?scene=${sel.value}`; };
  const go = (extra) => { location.search = `?scene=${sceneId}${extra}`; };

  let objects = [], composeSt = null, addHandler = null;
  const status = new Map(), rows = new Map(), metrics = new Map(), listeners = [];
  document.getElementById('addBtn').onclick = () => (addHandler ? addHandler() : go('&add=new'));

  const top = document.createElement('div'); top.className = 'row' + (currentRun ? '' : ' on');
  top.innerHTML = '<div class="ph">▦</div><div class="info"><div class="name">整个场景</div><div class="st"></div></div>';
  top.onclick = () => go(''); rowsEl.appendChild(top);

  async function act(path, what) {
    try { await post(path); } catch (e) { alert(`${what}失败：${e.message}`); }
    poll();
  }
  function actions(o, ph) {
    const box = document.createElement('div'); box.className = 'acts';
    const btn = (label, f, cls = '') => { const b = document.createElement('button'); b.className = 't ' + cls; b.textContent = label; b.onclick = (e) => { e.stopPropagation(); f(); }; box.appendChild(b); };
    const rerun = () => {
      const s = document.createElement('select');
      for (const [v, t] of [['', '重跑…'], ['gen', '换种子重新生成'], ['place', '重新摆放和删点']]) s.add(new Option(t, v));
      s.onclick = (e) => e.stopPropagation();
      s.onchange = () => {
        const v = s.value; s.value = ''; if (!v) return;
        act(`/api/run?run=${o.run}&from=${v}` + (v === 'gen' ? `&seed=${1 + Math.floor(Math.random() * 9999)}` : ''), '重跑');
      };
      box.appendChild(s);
    };
    const log = () => window.open(`${runDir(o.run)}/run.log`, '_blank');
    const del = () => { if (confirm(`从清单里去掉「${o.label || o.name}」？结果目录会留着，场景会重新合成。`)) act(`/api/remove?run=${o.run}`, '删除'); };
    if (ph.key === 'done') { btn('检查', () => go(`&run=${o.run}`), 'go'); rerun(); btn('删', del); }
    else if (ph.key === 'probe') { btn('看抠图', () => go(`&add=${o.run}`), 'go'); btn('删', del); }
    else if (ph.key === 'busy') { btn('停止', () => act(`/api/stop?run=${o.run}`, '停止')); btn('日志', log); }
    else if (ph.key === 'failed' || ph.key === 'stopped') { rerun(); btn('日志', log); btn('删', del); }
    else { btn('开始', () => act(`/api/run?run=${o.run}`, '开始'), 'go'); btn('删', del); }
    return box;
  }
  function makeRow(o) {
    const el = document.createElement('div'); el.className = 'row' + (o.run === currentRun ? ' on' : '');
    el.innerHTML = '<div class="ph">…</div><div class="info"><div class="name"></div><div class="st"></div><div class="bar"><i></i></div><div class="m"></div></div>';
    const r = { el, info: el.querySelector('.info'), thumb: el.querySelector('.ph'), thumbKey: null, actsKey: null, acts: null };
    el.onclick = () => { const ph = phase(status.get(o.run)); if (ph.key === 'done') go(`&run=${o.run}`); else if (ph.key === 'probe') go(`&add=${o.run}`); };
    rows.set(o.run, r); return r;
  }
  async function scores(o, st) {                       // IoU against the photo mask, worst leftover over the views
    const key = `${o.run}|${updatedMs(st)}`;
    if (metrics.has(key)) return metrics.get(key);
    const [ck, cmp] = await Promise.all([tryJSON(`${runDir(o.run)}/check.json`), tryJSON(`${runDir(o.run)}/compare.json`)]);
    const parts = [];
    if (ck && ck.silhouette_iou !== undefined) parts.push(`重合度 ${ck.silhouette_iou.toFixed(2)}`);
    if (cmp) { const l = Object.values(cmp).map((v) => v.leftover); if (l.length) parts.push(`残留 ≤${(100 * Math.max(...l)).toFixed(1)}%`); }
    metrics.set(key, parts.join(' · ')); return metrics.get(key);
  }
  function patch(o, r) {
    const st = status.get(o.run), ph = phase(st);
    r.info.querySelector('.name').innerHTML = '';
    r.info.querySelector('.name').append(o.label || o.name, Object.assign(document.createElement('span'), { className: 'en', textContent: ` ${o.prompt || ''}` }));
    const stEl = r.info.querySelector('.st'); stEl.className = 'st ' + ph.cls; stEl.textContent = ph.text;
    stEl.title = st && st.error ? st.error : '';
    const bar = r.info.querySelector('.bar'); bar.style.display = ph.frac !== undefined ? 'block' : 'none'; bar.firstChild.style.width = `${100 * (ph.frac || 0)}%`;
    const m = r.info.querySelector('.m');
    if (ph.key === 'done') scores(o, st).then((t) => { m.textContent = t; }); else m.textContent = st && st.gpu !== undefined && ph.key === 'busy' ? `显卡 ${st.gpu}` : '';
    const ak = `${ph.key}`;
    if (r.actsKey !== ak) { if (r.acts) r.acts.remove(); r.acts = actions(o, ph); r.info.appendChild(r.acts); r.actsKey = ak; }
    // thumbnail = the generation image with its mask, once the mask exists (redrawn when a new mask is made)
    const hasMask = !!st && (st.step === 'mask' ? st.state === 'done' : !['view', 'align', undefined].includes(st.step));
    const tk = hasMask ? `${o.run}|${ph.key === 'probe' ? updatedMs(st) : 'final'}` : null;
    if (tk && tk !== r.thumbKey) {
      r.thumbKey = tk; const c = document.createElement('canvas');
      drawMask(c, `${runDir(o.run)}/crop.jpg?t=${updatedMs(st)}`, `${runDir(o.run)}/mask_final.png?t=${updatedMs(st)}`, 128, true)
        .then(() => { r.thumb.replaceWith(c); r.thumb = c; }).catch(() => { /* no mask file: keep the placeholder */ });
    }
  }
  async function poll() {
    const sc = await tryJSON(`../scenes/${sceneId}.json`);
    if (sc && Array.isArray(sc.objects)) objects = sc.objects;
    const [sts, cst] = await Promise.all([Promise.all(objects.map((o) => tryJSON(`${runDir(o.run)}/status.json`))), tryJSON(`${composeDir(sceneId)}/status.json`)]);
    objects.forEach((o, k) => status.set(o.run, sts[k])); composeSt = cst;
    const keep = new Set(objects.map((o) => o.run));
    for (const [run, r] of rows) if (!keep.has(run)) { r.el.remove(); rows.delete(run); }
    for (const o of objects) { const r = rows.get(o.run) || makeRow(o); rowsEl.appendChild(r.el); patch(o, r); }
    const ts = top.querySelector('.st');
    ts.textContent = !sc ? '没找到物体清单（scenes/' + sceneId + '.json）' : `${objects.length} 个物体` +
      (composeSt && composeSt.state === 'running' ? ' · 合成中…' : composeSt && composeSt.state === 'failed' ? ' · 合成失败' : '');
    for (const f of listeners) f();
  }
  await poll(); setInterval(poll, 2000);
  return {
    sceneId, poll,
    objects: () => objects, status: (run) => status.get(run), composeStatus: () => composeSt,
    onPoll: (f) => listeners.push(f), setAddHandler: (f) => { addHandler = f; },
  };
}
