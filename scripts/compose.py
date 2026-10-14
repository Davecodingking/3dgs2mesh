# Experiment: put several replaced objects of one scene together, for the web viewer (viewer/index.html).
# Applies every run's carve to the original 3DGS (vote labels when the run has labels.npy, else the box in its
# place.json) and writes into <out_dir>:
#   scene.ply      the 3DGS with all those objects deleted
#   scene.splat    the same, pruned (alpha < 0.02 dropped) in the 32-byte .splat format the viewer loads
#   orig.splat     the untouched 3DGS, same format, to compare
#   <name>.glb     each run's placed mesh (gen/<name>_world.glb), in the 3DGS frame
#   objects.json   the meshes, and a point to orbit around (middle of the objects)
# usage: compose.py <scene.ply> <out_dir> <run_dir> [<run_dir> ...]
import os, sys, json, shutil, numpy as np, trimesh
from plyfile import PlyData, PlyElement
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carve import carve_drop
from carve_labels import label_drop
from splat import to_splat

scene_ply, out = sys.argv[1:3]; runs = sys.argv[3:]
os.makedirs(out, exist_ok=True)


v = PlyData.read(scene_ply)["vertex"].data
drop = np.zeros(len(v), bool); objects = []; centers = []
for r in runs:
    P = json.load(open(f"{r}/place.json"))
    d = label_drop(v, r)[0] if os.path.exists(f"{r}/labels.npy") else carve_drop(v, P["carve"])
    print(f"[compose] {P['name']}: carve removes {int(d.sum())} gaussians ({int((d & drop).sum())} already removed by others)")
    drop |= d
    shutil.copy(f"{r}/gen/{P['name']}_world.glb", f"{out}/{P['name']}.glb")
    m = trimesh.load(f"{out}/{P['name']}.glb", force="mesh"); centers.append(((m.bounds[0] + m.bounds[1]) / 2).tolist())
    objects.append(dict(name=P["name"], glb=f"{P['name']}.glb", center=centers[-1], run=os.path.basename(os.path.normpath(r))))
PlyData([PlyElement.describe(v[~drop], "vertex")]).write(f"{out}/scene.ply")
n1 = to_splat(v[~drop], f"{out}/scene.splat"); n0 = to_splat(v, f"{out}/orig.splat")
json.dump(dict(objects=objects, target=np.mean(centers, 0).tolist(), frame="OpenCV world: x right, y down, z forward; camera at origin"),
          open(f"{out}/objects.json", "w"), indent=1)
print(f"[compose] removed {int(drop.sum())} of {len(v)}; scene.splat {n1} splats ({n1 * 32 / 2**20:.1f} MB), "
      f"orig.splat {n0} splats -> {out}")
