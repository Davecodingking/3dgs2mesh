// The whole scene (what index.html used to be): the 3DGS with every replaced object deleted plus all the new meshes,
// or the original 3DGS to compare, from runs/<scene>_compose/. "+ 加物体" works here (add.js).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { scene, camera, world, controls, toThree, makeSplat, keepCamera } from './stage.js';
import { composeDir, tryJSON, post, updatedMs } from './api.js';
import { setupAdd } from './add.js';

export async function startScene(sceneId, side, addRun) {
  const dir = composeDir(sceneId), statusEl = document.getElementById('status');
  document.getElementById('sCompose').onclick = async () => {
    try { await post(`/api/compose?scene=${sceneId}`); side.poll(); } catch (e) { alert('合成失败：' + e.message); }
  };
  const objs = await tryJSON(`${dir}/objects.json`);
  if (!objs) {
    statusEl.textContent = `${sceneId} 还没有合成结果（runs/${sceneId}_compose/）。点上面的"重新合成"做一次，就能看场景、加物体了。`;
    return;
  }

  // start where you were in this scene, else at the panorama camera (a little behind it) facing the objects
  const target = toThree(objs.target);
  const home = () => {
    camera.fov = 70; camera.updateProjectionMatrix();
    camera.position.copy(target.clone().normalize().multiplyScalar(-0.05)); controls.target.copy(target); controls.update();
  };
  if (!keepCamera(`cam:scene:${sceneId}`)) home();
  addEventListener('keydown', (e) => { if ((e.key === 'r' || e.key === 'R') && e.target.tagName !== 'INPUT') home(); });
  document.getElementById('sHome').onclick = home;

  // 3DGS: after deletion (scene.splat) or original (orig.splat), each loaded the first time it is shown
  const gs = {};
  const layer = (k) => gs[k] || (gs[k] = makeSplat(`${dir}/${k === 'orig' ? 'orig' : 'scene'}.splat`).v);
  function setGS(k) {
    for (const v of Object.values(gs)) scene.remove(v);
    if (k !== 'none') scene.add(layer(k));
    document.querySelectorAll('[data-gs]').forEach((b) => b.classList.toggle('on', b.dataset.gs === k));
  }
  document.querySelectorAll('[data-gs]').forEach((b) => b.onclick = () => setGS(b.dataset.gs));
  setGS('after');

  // the new meshes, unlit: their texture is baked from the photo already
  const meshes = new THREE.Group(); world.add(meshes);
  const loader = new GLTFLoader();
  for (const o of objs.objects) {
    loader.load(`${dir}/${o.glb}`, (g) => {
      g.scene.traverse((m) => {
        if (!m.isMesh) return;
        const old = m.material;
        m.material = new THREE.MeshBasicMaterial({ map: old.map || null, color: old.map ? 0xffffff : (old.color || 0xffffff), side: THREE.DoubleSide });
      });
      meshes.add(g.scene);
    });
  }
  const mBtn = document.getElementById('sMesh');
  const showMeshes = (x) => { meshes.visible = x; mBtn.classList.toggle('on', x); mBtn.textContent = x ? '显示' : '隐藏'; };
  mBtn.onclick = () => showMeshes(!meshes.visible);

  // status line; when the composed scene changes underneath (an object finished, hand edits saved), say so
  const since = updatedMs(side.composeStatus());
  const names = objs.objects.map((o) => o.label || o.name).join('、');
  const line = () => {
    const cs = side.composeStatus();
    const news = cs && cs.state === 'done' && updatedMs(cs) > since ? '\n场景重新合成好了，按 F5 刷新就能看到' : cs && cs.state === 'running' ? '\n后台合成中…' : '';
    statusEl.textContent = `${sceneId}：${objs.objects.length} 个物体${names ? `（${names}）` : ''}${news}\n左键拖：转  滚轮：远近  右键拖：平移  R：回到拍摄位置`;
  };
  side.onPoll(line); line();

  const add = setupAdd({ sceneId, side, setGS, showMeshes });
  side.setAddHandler(() => add.enter('new'));
  if (addRun) add.enter(addRun);
}
