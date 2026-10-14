# Experiment: why does carving leave pieces of the original object? Takes the scene points that belong to the object as a
# camera saw it (inside that run's mask, from the front surface to `depth` x H behind it) and says which were deleted and,
# for the ones kept, which carve rule kept them. Carving itself never looks at the mask: it is a box around the placed mesh.
# usage: carve_report.py <mask_run> <place_run> <scene.ply> [depth_in_H=0.8] [min_opacity=0.1]
import sys, json, numpy as np
from PIL import Image
from plyfile import PlyData

mask_run, place_run, scene_ply = sys.argv[1:4]
depth = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
min_op = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
cam = json.load(open(f"{mask_run}/cam.json")); Rb = np.array(cam["Rb"]); f, S = cam["f"], cam["size"]
C0 = np.array(cam.get("C", [0.0, 0.0, 0.0]))
mask = np.array(Image.open(f"{mask_run}/mask_final.png").convert("L")) > 127
P = json.load(open(f"{place_run}/place.json")); cv = P["carve"]; H = P["unit"]
v = PlyData.read(scene_ply)["vertex"].data
X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
size = np.exp(np.max(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1), 1))

Q = (X - C0) @ Rb; z = Q[:, 2]; ok = (z > 0.05) & (op > min_op)
u = np.full(len(X), -1, np.int64); w = np.full(len(X), -1, np.int64)
u[ok] = np.floor(f * Q[ok, 0] / z[ok] + S / 2); w[ok] = np.floor(f * Q[ok, 1] / z[ok] + S / 2)
ok &= (u >= 0) & (u < S) & (w >= 0) & (w < S)
inm = np.zeros(len(X), bool); inm[ok] = mask[w[ok], u[ok]]
hist, edges = np.histogram(z[inm], bins=np.arange(0.1 * H, 6 * H, 0.05 * H))
Zf = edges[int(np.argmax(hist >= 0.25 * hist.max()))] + 0.025 * H
obj = inm & (z > Zf - 0.08 * H) & (z < Zf + depth * H)

ca, sa = np.cos(cv["phi"]), np.sin(cv["phi"])
lo, hi, pad = np.array(cv["lo"]), np.array(cv["hi"]), np.array(cv["pad"]); ext = hi - lo
a, b, h = X[:, 0] * ca + X[:, 2] * sa, -X[:, 0] * sa + X[:, 2] * ca, -X[:, 1]
in_ab = (a > lo[0] - pad[0] * ext[0]) & (a < hi[0] + pad[0] * ext[0]) & (b > lo[1] - pad[1] * ext[1]) & (b < hi[1] + pad[1] * ext[1])
below_top = h < hi[2] - cv["top_margin"]; above_floor = h > cv["floor_h"]; above_lo = h > lo[2] - pad[2] * ext[2]
drop = in_ab & below_top & above_floor & above_lo

n = int(obj.sum()); kept = obj & ~drop
print(f"[report] object points (mask {mask_run.split('/')[-1]}, opacity > {min_op}, front {Zf:.3f} .. +{depth} H): {n}")
print(f"[report] deleted {int((obj & drop).sum())} ({100 * (obj & drop).sum() / n:.0f}%), kept {int(kept.sum())} ({100 * kept.sum() / n:.0f}%)")
reasons = [("above the new mesh's top (kept on purpose: things standing on it)", kept & in_ab & ~below_top),
           ("in the floor layer (lower than floor + 3% H)", kept & in_ab & below_top & ~above_floor),
           ("outside the box sideways", kept & ~in_ab)]
for name, m in reasons:
    k = int(m.sum())
    if k: print(f"[report]   {k:6d} ({100 * k / n:4.1f}%)  {name}; their height above floor {np.median(h[m]) - (-P['metrics']['floor_y']):+.3f} (median), "
                f"big blurry ones (size > 0.02) {100 * (size[m] > 0.02).mean():.0f}%")
nf = obj & above_floor & in_ab
print(f"[report] heights above floor: original object top (98th pct of its points) {np.percentile(h[nf], 98) + P['metrics']['floor_y']:.3f}, "
      f"new mesh top {hi[2] + P['metrics']['floor_y']:.3f} (H = {H:.3f})")
