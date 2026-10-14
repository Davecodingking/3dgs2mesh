# Experiment: data for the 3D check page (viewer/debug.html), for the gaussians near the object (vote.py's radius):
#   debug/points.bin   float32 xyz x n, then uint8 x n x 4: r g b, class
#                      class: 0 deleted, 1 voted object but kept (floor layer), 2 unseen and kept, 3 other (kept);
#                      hand edits (manual.json) already applied, as carve_labels does
#   debug/votes.bin    uint32 seen bits x n, then uint32 inside bits x n (bit k = view k), from vote.py
#   debug/index.bin    uint32 x n: each point's index in the scene file (the page saves hand edits with these)
#   debug/before.splat, after.splat   the 3DGS around the object (radius max(1.0, 4 x half width)) before and after
#                      this object's deletion, for the page's 3D scene layer
#   debug/before_index.bin  uint32 per splat of before.splat: its scene index (the page rebuilds "after" from before.splat
#                      by dropping what is deleted, without asking the server)
#   debug/mesh_vox.bin uint8 x (nx*ny*nz): the placed mesh as a solid (voxels, grown by one, holes filled), for the page's
#                      mesh eraser preview; grid in meta.vox (origin of index 0, pitch, shape), index = round((p - origin) / pitch)
#   debug/meta.json    n, object centre, views (camera, name, has a mask, weight, image), the deletion closure (outline
#                      x/z, floor y, bottom / top heights), the mesh and its centre, counts per class
# Writes views/<k>/overlay.jpg for given views too (rendered views have one from mask_views.py).
# usage: debug_export.py <run> <scene.ply> <name>
import os, sys, json, numpy as np, cv2
from plyfile import PlyData
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carve_labels import label_drop, closure
import trimesh
from splat import to_splat

run, scene_ply, name = sys.argv[1:4]
meta = json.load(open(f"{run}/views.json"))
v = PlyData.read(scene_ply)["vertex"].data
X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
labels = np.load(f"{run}/labels.npy"); drop = label_drop(v, run)[0]
vt = np.load(f"{run}/votes.npz"); near = vt["near"].astype(np.int64)
cls = np.full(len(X), 3, np.uint8); cls[labels == 2] = 2; cls[labels == 1] = 1; cls[drop] = 0
rgb = np.clip(0.5 + 0.28209479177387814 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
att = np.zeros((len(near), 4), np.uint8); att[:, :3] = np.round(rgb[near] * 255); att[:, 3] = cls[near]
os.makedirs(f"{run}/debug", exist_ok=True)
with open(f"{run}/debug/points.bin", "wb") as fh:
    fh.write(X[near].astype(np.float32).tobytes()); fh.write(att.tobytes())
with open(f"{run}/debug/votes.bin", "wb") as fh:
    fh.write(vt["seen"].astype(np.uint32).tobytes()); fh.write(vt["inside"].astype(np.uint32).tobytes())
near.astype(np.uint32).tofile(f"{run}/debug/index.bin")
crop = np.linalg.norm(X - np.array(meta["center"]), axis=1) < max(1.0, 4 * meta["radius"])
crop &= 1 / (1 + np.exp(-v["opacity"].astype(np.float64))) >= 0.02                    # what to_splat would keep anyway
nb = to_splat(v[crop], f"{run}/debug/before.splat", 0.0); na = to_splat(v[crop & ~drop], f"{run}/debug/after.splat", 0.0)
np.nonzero(crop)[0].astype(np.uint32).tofile(f"{run}/debug/before_index.bin")
from scipy import ndimage
_m = trimesh.load(f"{run}/gen/{name}_world.glb", force="mesh")
_g = trimesh.Trimesh(np.asarray(_m.vertices), np.asarray(_m.faces), process=True); _vx = _g.voxelized(_g.extents.max() / 160)
_solid = ndimage.binary_fill_holes(ndimage.binary_dilation(np.pad(_vx.matrix, 1), iterations=1))
_solid.astype(np.uint8).tofile(f"{run}/debug/mesh_vox.bin")
vox = dict(pitch=float(_vx.pitch.max() if np.ndim(_vx.pitch) else _vx.pitch), shape=list(_solid.shape))
vox["origin"] = (np.asarray(_vx.indices_to_points(np.array([[0, 0, 0]])), float)[0] - vox["pitch"]).tolist()  # index 0 after padding
views = []
for vw in meta["views"]:
    d = f"{run}/views/{vw['k']}"
    if vw.get("mask") and not os.path.exists(f"{d}/overlay.jpg"):
        img = cv2.imread(f"{d}/frames/0000.jpg"); m = cv2.imread(f"{d}/mask.png", 0) > 127
        img[m] = (0.5 * img[m] + 0.5 * np.array([60, 200, 90])).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(img, cs, -1, (0, 255, 255), 2)
        cv2.imwrite(f"{d}/overlay.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    views.append(dict(k=vw["k"], name=vw["name"], C=vw["C"], Rb=vw["Rb"], f=vw["f"], size=vw["size"], mask=bool(vw.get("mask")),
                      weight=float(vw.get("weight", 1.0)), image=f"../views/{vw['k']}/{'overlay.jpg' if vw.get('mask') else 'frames/0000.jpg'}"))
counts = {k: int((cls[near] == i).sum()) for i, k in enumerate(("deleted", "object_kept", "unseen_kept", "other"))}
_, poly, bottom, top, _ = closure(v, run); P = json.load(open(f"{run}/place.json"))
clo = dict(poly=poly.tolist(), floor_y=P["metrics"]["floor_y"], bottom=bottom, top=top)
_mv = trimesh.load(f"{run}/gen/{name}_world.glb", force="mesh").vertices; mc = ((_mv.min(0) + _mv.max(0)) / 2).tolist()
json.dump(dict(n=int(len(near)), center=meta["center"], radius=meta["radius"], views=views,
               closure=clo, mesh=f"../gen/{name}_world.glb", mesh_center=mc, vox=vox, unit=P["unit"], name=name, scene_ply=os.path.abspath(scene_ply),
               run=os.path.basename(os.path.normpath(run)), counts=counts), open(f"{run}/debug/meta.json", "w"), indent=1)
print(f"[debug_export] {len(near)} points near the object {counts} -> {run}/debug/")
