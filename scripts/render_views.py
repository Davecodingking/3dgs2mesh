# Experiment: several views of one object rendered from the 3DGS (step 1 of the multi-view flow).
# The object's seed points are the scene points inside the source run's mask (front surface + a little behind it). Cameras
# look at their centre from the panorama camera's side (the back of an object was never seen, renders there are mush):
# azimuth / height pairs in VIEWS, distance from the object's width. For each view writes
#   <out>/views/<k>/frames/0000.jpg (the render, SAM3 reads a folder), depth.npy (expected depth), cam.json, seed_px.npy
# (seed points projected into this view: mask_views.py keeps the SAM3 instance that covers most of them), and
# <out>/views.json + <out>/seed.npy for the later steps.
# Also adds "given" views that already have a mask: the source run's own camera (its crop.jpg / mask_final.png; the photo
# for a panorama crop) and the same from every extra run given (e.g. the panorama crop of an object generated from a
# rendered view). Only their depth is rendered here. They weigh double in the vote and the fit: they are the sharpest.
# usage: render_views.py <src_run> <scene.ply> <out_run> [extra_run ...]
import os, sys, json, numpy as np, torch, cv2
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization

src, scene_ply, out = sys.argv[1:4]; extra = sys.argv[4:]
import shutil
VIEWS = [(-45, "mid"), (0, "mid"), (45, "mid"), (-20, "high"), (20, "high"), (0, "low")]   # azimuth from the pano side
S, FOV = 1024, 60.0; f = S / 2 / np.tan(np.radians(FOV) / 2)

cam0 = json.load(open(f"{src}/cam.json")); Rb0 = np.array(cam0["Rb"]); f0, S0 = cam0["f"], cam0["size"]
C0 = np.array(cam0.get("C", [0.0, 0.0, 0.0]))
mask0 = np.array(Image.open(f"{src}/mask_final.png").convert("L")) > 127
v = PlyData.read(scene_ply)["vertex"].data
P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
H = max(0.05, float(np.percentile(P[op > 0.4, 1], 97)))     # camera height above the floor (place.py's unit)
floor_y = H                                                  # the panorama camera is at y = 0, y is down

Q = (P - C0) @ Rb0; z = Q[:, 2]; ok = (z > 0.05) & (op > 0.4)
u = np.full(len(P), -1, np.int64); w = np.full(len(P), -1, np.int64)
u[ok] = np.floor(f0 * Q[ok, 0] / z[ok] + S0 / 2); w[ok] = np.floor(f0 * Q[ok, 1] / z[ok] + S0 / 2)
ok &= (u >= 0) & (u < S0) & (w >= 0) & (w < S0)
inm = np.zeros(len(P), bool); inm[ok] = mask0[w[ok], u[ok]]
hist, edges = np.histogram(z[inm], bins=np.arange(0.1 * H, 6 * H, 0.05 * H))
Zf = edges[int(np.argmax(hist >= 0.25 * hist.max()))] + 0.025 * H
seed = P[inm & (z > Zf - 0.08 * H) & (z < Zf + 0.8 * H)]
center = np.median(seed, 0)
r = 0.5 * max(np.percentile(seed[:, k], 98) - np.percentile(seed[:, k], 2) for k in (0, 2))
dist = float(np.clip(2.6 * r + 0.15, 0.4, 1.2))
heights = dict(low=0.12, mid=0.5 * H, high=H)
base = C0 - center; base[1] = 0; base /= np.linalg.norm(base)

g = lambda *k: torch.tensor(np.stack([v[n] for n in k], 1).astype(np.float32), device="cuda")
qt = g("rot_0", "rot_1", "rot_2", "rot_3")
GS = dict(means=g("x", "y", "z"), quats=qt / qt.norm(dim=1, keepdim=True), scales=torch.exp(g("scale_0", "scale_1", "scale_2")),
          opacities=torch.sigmoid(g("opacity")[:, 0]), colors=(0.5 + 0.28209479177387814 * g("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0, 1))
K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]], np.float32)
views = []
for k, (az, hn) in enumerate(VIEWS):
    t = np.radians(az); c, s = np.cos(t), np.sin(t)
    side = np.array([c * base[0] + s * base[2], 0.0, -s * base[0] + c * base[2]])
    C = center + dist * side; C[1] = floor_y - heights[hn]
    zc = center - C; zc /= np.linalg.norm(zc); xc = np.cross([0.0, 1.0, 0.0], zc); xc /= np.linalg.norm(xc)
    Rb = np.stack([xc, np.cross(zc, xc), zc], 1)
    view = np.eye(4, dtype=np.float32); view[:3, :3] = Rb.T; view[:3, 3] = -Rb.T @ C
    img, alpha, _ = rasterization(**GS, viewmats=torch.tensor(view, device="cuda")[None], Ks=torch.tensor(K, device="cuda")[None],
                                  width=S, height=S, near_plane=0.01, render_mode="RGB+ED")
    d = f"{out}/views/{k}"; os.makedirs(f"{d}/frames", exist_ok=True)
    cv2.imwrite(f"{d}/frames/0000.jpg", (img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(f"{d}/depth.npy", img[0, ..., 3].cpu().numpy().astype(np.float32))
    cam = dict(fov=FOV, size=S, f=f, Rb=Rb.tolist(), C=C.tolist(), azimuth=az, height=heights[hn], name=f"{az:+d} {hn}")
    json.dump(cam, open(f"{d}/cam.json", "w"), indent=1)
    sq = (seed - C) @ Rb; sp = np.stack([f * sq[:, 0] / sq[:, 2] + S / 2, f * sq[:, 1] / sq[:, 2] + S / 2], 1)
    np.save(f"{d}/seed_px.npy", sp[sq[:, 2] > 0.05].astype(np.float32))
    views.append(dict(k=k, **cam))
for gk, g_run in enumerate([src] + extra):                  # given views: camera and mask exist, render the depth
    gc = json.load(open(f"{g_run}/cam.json")); Rb = np.array(gc["Rb"]); C = np.array(gc.get("C", [0.0, 0.0, 0.0]))
    Sg, fg = int(gc["size"]), float(gc["f"]); Kg = np.array([[fg, 0, Sg / 2], [0, fg, Sg / 2], [0, 0, 1]], np.float32)
    view = np.eye(4, dtype=np.float32); view[:3, :3] = Rb.T; view[:3, 3] = -Rb.T @ C
    img, _, _ = rasterization(**GS, viewmats=torch.tensor(view, device="cuda")[None], Ks=torch.tensor(Kg, device="cuda")[None],
                              width=Sg, height=Sg, near_plane=0.01, render_mode="RGB+ED")
    k = len(views); d = f"{out}/views/{k}"; os.makedirs(f"{d}/frames", exist_ok=True)
    shutil.copy(f"{g_run}/crop.jpg", f"{d}/frames/0000.jpg"); shutil.copy(f"{g_run}/mask_final.png", f"{d}/mask.png")
    np.save(f"{d}/depth.npy", img[0, ..., 3].cpu().numpy().astype(np.float32))
    cam = dict(fov=gc["fov"], size=Sg, f=fg, Rb=Rb.tolist(), C=C.tolist(), name=f"given {os.path.basename(os.path.normpath(g_run))}",
               given=True, weight=2.0, mask=True)
    json.dump(cam, open(f"{d}/cam.json", "w"), indent=1)
    sq = (seed - C) @ Rb; sp = np.stack([fg * sq[:, 0] / sq[:, 2] + Sg / 2, fg * sq[:, 1] / sq[:, 2] + Sg / 2], 1)
    np.save(f"{d}/seed_px.npy", sp[sq[:, 2] > 0.05].astype(np.float32))
    views.append(dict(k=k, **cam))
np.save(f"{out}/seed.npy", seed.astype(np.float32))
json.dump(dict(source=os.path.basename(os.path.normpath(src)), center=center.tolist(), radius=r, dist=dist, unit=H,
               floor_y=floor_y, views=views), open(f"{out}/views.json", "w"), indent=1)
print(f"[views] {len(seed)} seed points, centre {center.round(3)}, half width {r:.3f}, camera distance {dist:.2f}, "
      f"{len(views)} views -> {out}/views/")
