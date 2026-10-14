# Experiment: the check page's mesh eraser. The placed mesh, scaled per axis about its centre and moved, is a 3D brush:
# every point near the object that falls inside it (above the floor layer, which always stays) is added to manual.json
# "delete" and taken out of "keep". Inside = the mesh voxelised (pitch = its longest side / 160), grown by one voxel,
# holes filled, so a mesh that is not quite closed still erases its inside.
# usage: mesh_erase.py <run> '<json {"scale": [sx, sy, sz], "offset": [dx, dy, dz]}>'   (world axes, y is down)
import os, sys, json, numpy as np, trimesh
from scipy import ndimage

run, params = sys.argv[1], json.loads(sys.argv[2])
meta = json.load(open(f"{run}/debug/meta.json")); name = meta["name"]
P = json.load(open(f"{run}/place.json")); floor, H = P["metrics"]["floor_y"], P["unit"]
m = trimesh.load(f"{run}/gen/{name}_world.glb", force="mesh")
V = np.asarray(m.vertices, float); c = (V.min(0) + V.max(0)) / 2
geo = trimesh.Trimesh(c + (V - c) * np.array(params["scale"], float) + np.array(params["offset"], float), np.asarray(m.faces), process=True)
vox = geo.voxelized(geo.extents.max() / 160)
mat = ndimage.binary_fill_holes(ndimage.binary_dilation(np.pad(vox.matrix, 1), iterations=1))

n = meta["n"]
pos = np.fromfile(f"{run}/debug/points.bin", np.float32, count=n * 3).reshape(n, 3).astype(np.float64)
idx = np.fromfile(f"{run}/debug/index.bin", np.uint32).astype(np.int64)
ind = np.asarray(vox.points_to_indices(pos), np.int64) + 1                    # +1: the padding
ok = np.all((ind >= 0) & (ind < np.array(mat.shape)), axis=1)
inside = np.zeros(n, bool); inside[ok] = mat[ind[ok, 0], ind[ok, 1], ind[ok, 2]]
inside &= (floor - pos[:, 1]) > 0.03 * H

man = json.load(open(f"{run}/manual.json")) if os.path.exists(f"{run}/manual.json") else {}
new = set(idx[inside].tolist())
man = dict(delete=sorted(set(man.get("delete", [])) | new), keep=sorted(set(man.get("keep", [])) - new))
json.dump(man, open(f"{run}/manual.json", "w"))
print(f"[mesh_erase] scale {params['scale']} offset {params['offset']}: {int(inside.sum())} points inside the mesh brush "
      f"-> manual.json (delete {len(man['delete'])}, keep {len(man['keep'])})")
