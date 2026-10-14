// The 3D stage every mode draws on: renderer, camera, orbit controls, and `world`, the group for everything that is in
// the OpenCV world of the scenes (x right, y down, z forward, panorama camera at the origin). three.js is y up, so
// `world` is turned 180 deg about x. The canvas fills the window right of the object list.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as GS from '@mkkellogg/gaussian-splats-3d';

export const el = document.getElementById('canvas');
export const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
el.appendChild(renderer.domElement);
export const scene = new THREE.Scene(); scene.background = new THREE.Color(0x111111);
export const camera = new THREE.PerspectiveCamera(60, 1, 0.005, 50);
export const world = new THREE.Group(); world.rotation.x = Math.PI; scene.add(world); world.updateMatrixWorld();
export const controls = new OrbitControls(camera, renderer.domElement);
export const toThree = (p) => new THREE.Vector3(p[0], -p[1], -p[2]);

export const size = () => ({ w: el.clientWidth, h: el.clientHeight });
export const local = (ev) => { const r = el.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
const onResize = [];
export const resized = (f) => onResize.push(f);
function resize() {
  const { w, h } = size(); camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
  for (const f of onResize) f();
}
addEventListener('resize', resize); resize();

const hooks = [];
export const onFrame = (f) => hooks.push(f);
renderer.setAnimationLoop(() => { for (const f of hooks) f(); controls.update(); renderer.render(scene, camera); });

// a 3DGS layer; .splat files and Blob URLs (no extension) need the format said explicitly
export function makeSplat(url, format) {
  const v = new GS.DropInViewer({ sharedMemoryForWorkers: false, gpuAcceleratedSort: false });
  const opt = { rotation: [1, 0, 0, 0], showLoadingUI: false, splatAlphaRemovalThreshold: 5 };
  if (format !== undefined) opt.format = format;
  return { v, ready: v.addSplatScene(url, opt) };
}
export const SplatFormat = GS.SceneFormat;

// the camera survives a reload of the same page (key: per object / per scene)
export function keepCamera(key) {
  let restored = false;
  try {
    const c = JSON.parse(sessionStorage.getItem(key) || 'null');
    if (c) { camera.fov = c.fov; camera.updateProjectionMatrix(); camera.position.fromArray(c.p); controls.target.fromArray(c.t); controls.update(); restored = true; }
  } catch (e) { /* no storage */ }
  controls.addEventListener('change', () => {
    try { sessionStorage.setItem(key, JSON.stringify({ p: camera.position.toArray(), t: controls.target.toArray(), fov: camera.fov })); } catch (e) { /* ignore */ }
  });
  return restored;
}

// top of the free area below the toolbar (it wraps to more lines on narrow windows)
export const belowBar = () => document.getElementById('bar').getBoundingClientRect().bottom + 8;
