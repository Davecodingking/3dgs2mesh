# Experiment: which 3DGS points belong to the object (step 3 of the multi-view flow): every view with a mask votes.
# A point is "seen" in a view when it lands in the image in front of the camera and is not hidden (its depth is at most
# the rendered depth there plus a tolerance); "inside" when it lands in the mask. An elongated gaussian is judged by three
# points along its longest axis (centre and both ends at 1.5 sigma), inside when at least two of them are.
#   object   seen in >= 1 view and inside in >= 60% of the views that see it (weighted: given views count double)
#   unseen   seen in no view (the underside of a table top, the inside of a base); carving falls back to the mesh box
#   other    everything else (floor, things standing on it, neighbours)
# Only points near the object are looked at (3 x its half width + 0.2 around its centre); the rest are "other".
# Masks are grown by DILATE px first: seen almost edge-on (a table top from the mid-height views) the object is a thin band
# and a mask that is a few px short there out-voted the high views (A01 table: a film of the old top survived).
# Writes <run>/labels.npy (uint8 per gaussian of the scene file: 0 other, 1 object, 2 unseen) and <run>/votes.npz
# (the near points' scene indices, and per view bit k: seen / inside), which the 3D check page shows view by view.
# usage: vote.py <run> <scene.ply>
import sys, json, numpy as np, cv2
from PIL import Image
from plyfile import PlyData

run, scene_ply = sys.argv[1:3]
DILATE = 6
meta = json.load(open(f"{run}/views.json")); center = np.array(meta["center"]); r = meta["radius"]
v = PlyData.read(scene_ply)["vertex"].data
X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
near = np.nonzero(np.linalg.norm(X - center, axis=1) < 3 * r + 0.2)[0]
Xn = X[near]
q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1)[near].astype(np.float64)
q /= np.linalg.norm(q, axis=1, keepdims=True); w_, x_, y_, z_ = q.T
R = np.stack([np.stack([1 - 2 * (y_ ** 2 + z_ ** 2), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)], 1),
              np.stack([2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ ** 2 + z_ ** 2), 2 * (y_ * z_ - w_ * x_)], 1),
              np.stack([2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ ** 2 + y_ ** 2)], 1)], 1)   # rows
sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)[near].astype(np.float64))
ax = np.argmax(sc, 1); major = R[np.arange(len(near)), :, ax] * (1.5 * sc[np.arange(len(near)), ax])[:, None]
samples = [Xn, Xn + major, Xn - major]                     # centre, both ends of the longest axis

seen_n = np.zeros(len(near)); in_n = np.zeros(len(near)); used = []
seen_bits = np.zeros(len(near), np.uint32); in_bits = np.zeros(len(near), np.uint32)
for vw in meta["views"]:
    if not vw.get("mask"): continue
    d = f"{run}/views/{vw['k']}"; used.append(vw["k"])
    Rb, C, f, S = np.array(vw["Rb"]), np.array(vw["C"]), vw["f"], vw["size"]
    D = np.load(f"{d}/depth.npy"); M = np.array(Image.open(f"{d}/mask.png").convert("L")) > 127
    M = cv2.dilate(M.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1,) * 2)) > 0
    def look(P):
        Q = (P - C) @ Rb; z = Q[:, 2]; ok = z > 0.05
        u = np.full(len(P), -1, np.int64); w = np.full(len(P), -1, np.int64)
        u[ok] = np.floor(f * Q[ok, 0] / z[ok] + S / 2); w[ok] = np.floor(f * Q[ok, 1] / z[ok] + S / 2)
        ok &= (u >= 0) & (u < S) & (w >= 0) & (w < S)
        return z, u, w, ok
    z, u, w, ok = look(Xn)
    seen = np.zeros(len(near), bool); seen[ok] = z[ok] <= D[w[ok], u[ok]] * 1.02 + 0.02
    votes = np.zeros(len(near), np.int32)
    for P in samples:
        _, us, ws, oks = look(P); ins = np.zeros(len(near), bool); ins[oks] = M[ws[oks], us[oks]]; votes += ins
    wt = float(vw.get("weight", 1.0)); seen_n += wt * seen; in_n += wt * (seen & (votes >= 2))
    seen_bits |= seen.astype(np.uint32) << vw["k"]; in_bits |= (votes >= 2).astype(np.uint32) << vw["k"]
labels = np.zeros(len(X), np.uint8)
lab = np.where(seen_n == 0, 2, np.where(in_n >= 0.6 * seen_n, 1, 0)).astype(np.uint8)
labels[near] = lab
np.save(f"{run}/labels.npy", labels)
np.savez_compressed(f"{run}/votes.npz", near=near.astype(np.uint32), seen=seen_bits, inside=in_bits, views=np.array(used))
print(f"[vote] views used {used}; near the object {len(near)} points: object {int((lab == 1).sum())}, "
      f"unseen {int((lab == 2).sum())}, other {int((lab == 0).sum())} -> {run}/labels.npy")
