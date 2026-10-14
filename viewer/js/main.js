// The tool page. Left: the scene's object list (sidebar.js). Right, chosen by the URL:
//   ?scene=A01                 the whole scene (scene.js), where objects are added (add.js)
//   ?scene=A01&add=new|<run>   adding an object / looking at a new object's mask
//   ?scene=A01&run=<run>       one object's check (check.js); old debug.html?run=<run> links land here too
import { tryJSON } from './api.js';
import { startSidebar } from './sidebar.js';

const q = new URLSearchParams(location.search);
const run = q.get('run');
const statusEl = document.getElementById('status');
try {
  const scenes = (await tryJSON('../scenes/index.json')) || [];
  const sceneId = q.get('scene') || (run ? run.split('_')[0] : scenes.length ? scenes[0].id : 'A01');
  document.body.dataset.mode = run ? 'check' : 'scene';
  const side = await startSidebar(sceneId, scenes, run);
  if (run) await (await import('./check.js')).startCheck(run, side);
  else await (await import('./scene.js')).startScene(sceneId, side, q.get('add'));
} catch (e) {
  statusEl.textContent = '出错：' + e.message; console.error(e);
}
