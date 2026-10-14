# Step debug: data for the 3D check page (viewer/debug.html), for the gaussians near the object (vote's radius):
#   debug/points.bin   float32 xyz x n, then uint8 x n x 4: r g b, class
#                      class: 0 deleted, 1 voted object but kept (floor layer), 2 unseen and kept, 3 other (kept);
#                      hand edits (manual.json) already applied, as carve does
#   debug/votes.bin    uint32 seen bits x n, then uint32 inside bits x n (bit k = view k), from vote
#   debug/index.bin    uint32 x n: each point's index in the scene file (the page saves hand edits with these)
#   debug/before.splat, after.splat   the 3DGS around the object (radius max(1.0, 4 x half width)) before and after
#                      this object's deletion, for the page's 3D scene layer
#   debug/before_index.bin  uint32 per splat of before.splat: its scene index (the page rebuilds "after" from before.splat
#                      by dropping what is deleted, without asking the server)
#   debug/mesh_vox.bin uint8 x (nx*ny*nz): the placed mesh as a solid (voxels, grown by one, holes filled), for the page's
#                      mesh eraser; grid in meta.vox (origin of index 0, pitch, shape), index = round((p - origin) / pitch)
#   debug/meta.json    n, object centre, views (camera, name, has a mask, weight, image), the deletion closure (outline
#                      x/z, floor y, bottom / top heights), the mesh and its centre, counts per class
# Writes views/<k>/overlay.jpg for given views too (the others have one from vmask).
import os, json, numpy as np, cv2
from .common import read_ply, xyz, opacity, to_splat, SH0
from .carve import label_drop, closure


def run(c):
    import trimesh
    from scipy import ndimage
    run_dir, name = c.dir, c.name
    meta = json.load(open(f"{run_dir}/views.json"))
    v = read_ply(c.ply); X = xyz(v)
    labels = np.load(f"{run_dir}/labels.npy"); drop = label_drop(v, run_dir)[0]
    vt = np.load(f"{run_dir}/votes.npz"); near = vt["near"].astype(np.int64)
    cls = np.full(len(X), 3, np.uint8); cls[labels == 2] = 2; cls[labels == 1] = 1; cls[drop] = 0
    rgb = np.clip(0.5 + SH0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
    att = np.zeros((len(near), 4), np.uint8); att[:, :3] = np.round(rgb[near] * 255); att[:, 3] = cls[near]
    os.makedirs(f"{run_dir}/debug", exist_ok=True)
    with open(f"{run_dir}/debug/points.bin", "wb") as fh:
        fh.write(X[near].astype(np.float32).tobytes()); fh.write(att.tobytes())
    with open(f"{run_dir}/debug/votes.bin", "wb") as fh:
        fh.write(vt["seen"].astype(np.uint32).tobytes()); fh.write(vt["inside"].astype(np.uint32).tobytes())
    near.astype(np.uint32).tofile(f"{run_dir}/debug/index.bin")
    crop = np.linalg.norm(X - np.array(meta["center"]), axis=1) < max(1.0, 4 * meta["radius"])
    crop &= opacity(v) >= 0.02                                # what to_splat would keep anyway
    to_splat(v[crop], f"{run_dir}/debug/before.splat", 0.0); to_splat(v[crop & ~drop], f"{run_dir}/debug/after.splat", 0.0)
    np.nonzero(crop)[0].astype(np.uint32).tofile(f"{run_dir}/debug/before_index.bin")

    mesh = trimesh.load(f"{run_dir}/gen/{name}_world.glb", force="mesh")
    geo = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=True); vx = geo.voxelized(geo.extents.max() / 160)
    solid = ndimage.binary_fill_holes(ndimage.binary_dilation(np.pad(vx.matrix, 1), iterations=1))
    solid.astype(np.uint8).tofile(f"{run_dir}/debug/mesh_vox.bin")
    vox = dict(pitch=float(vx.pitch.max() if np.ndim(vx.pitch) else vx.pitch), shape=list(solid.shape))
    vox["origin"] = (np.asarray(vx.indices_to_points(np.array([[0, 0, 0]])), float)[0] - vox["pitch"]).tolist()  # index 0 after padding

    views = []
    for vw in meta["views"]:
        d = f"{run_dir}/views/{vw['k']}"
        if vw.get("mask") and not os.path.exists(f"{d}/overlay.jpg"):
            img = cv2.imread(f"{d}/frames/0000.jpg"); m = cv2.imread(f"{d}/mask.png", 0) > 127
            img[m] = (0.5 * img[m] + 0.5 * np.array([60, 200, 90])).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(img, cs, -1, (0, 255, 255), 2)
            cv2.imwrite(f"{d}/overlay.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        views.append(dict(k=vw["k"], name=vw["name"], C=vw["C"], Rb=vw["Rb"], f=vw["f"], size=vw["size"], mask=bool(vw.get("mask")),
                          weight=float(vw.get("weight", 1.0)), image=f"../views/{vw['k']}/{'overlay.jpg' if vw.get('mask') else 'frames/0000.jpg'}"))
    counts = {k: int((cls[near] == i).sum()) for i, k in enumerate(("deleted", "object_kept", "unseen_kept", "other"))}
    _, poly, bottom, top, _ = closure(v, run_dir); P = json.load(open(f"{run_dir}/place.json"))
    clo = dict(poly=poly.tolist(), floor_y=P["metrics"]["floor_y"], bottom=bottom, top=top)
    mv = np.asarray(mesh.vertices); mc = ((mv.min(0) + mv.max(0)) / 2).tolist()
    json.dump(dict(n=int(len(near)), center=meta["center"], radius=meta["radius"], views=views,
                   closure=clo, mesh=f"../gen/{name}_world.glb", mesh_center=mc, vox=vox, unit=P["unit"], name=name,
                   scene_ply=os.path.abspath(c.ply), run=c.run, counts=counts), open(f"{run_dir}/debug/meta.json", "w"), indent=1)
    print(f"[debug] {len(near)} points near the object {counts} -> {run_dir}/debug/")
