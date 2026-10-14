# Experiment: a better view of an object for generation, rendered from the 3DGS itself. From the panorama centre some
# objects are seen too steeply (A01 coffee table: SAM 3D makes a flat disc without its base, 4 seeds out of 4).
# Finds the object's centre from the scene points inside an existing run's mask, puts a camera lower and further out on the
# same side as the panorama camera, renders the 3DGS there and writes a new run dir: crop.jpg (the render), frames/0000.jpg
# and cam.json (same fields as crop.py plus the camera position C), so mask / gen / place / check run on it unchanged.
# usage: novel_view.py <src_run> <scene.ply> <out_run> <height_above_floor> <distance> [fov=60] [turn_deg=0]
import os, sys, json, numpy as np, torch, cv2
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization

src, scene_ply, out = sys.argv[1:4]
height, dist = float(sys.argv[4]), float(sys.argv[5])
fov = float(sys.argv[6]) if len(sys.argv) > 6 else 60.0
turn = np.radians(float(sys.argv[7])) if len(sys.argv) > 7 else 0.0
S = 1024; f = S / 2 / np.tan(np.radians(fov) / 2)

cam0 = json.load(open(f"{src}/cam.json")); Rb0 = np.array(cam0["Rb"]); f0 = cam0["f"]; S0 = cam0["size"]
C0 = np.array(cam0.get("C", [0.0, 0.0, 0.0]))
mask = np.array(Image.open(f"{src}/mask_final.png").convert("L")) > 127
v = PlyData.read(scene_ply)["vertex"].data
P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
Q = (P - C0) @ Rb0; ok = (Q[:, 2] > 0.05) & (op > 0.4)
u = np.full(len(P), -1, np.int64); w = np.full(len(P), -1, np.int64)
u[ok] = np.floor(f0 * Q[ok, 0] / Q[ok, 2] + S0 / 2); w[ok] = np.floor(f0 * Q[ok, 1] / Q[ok, 2] + S0 / 2)
ok &= (u >= 0) & (u < S0) & (w >= 0) & (w < S0)
inm = np.zeros(len(P), bool); inm[ok] = mask[w[ok], u[ok]]
obj = P[inm]; zf = np.percentile(Q[inm, 2], 10); obj = obj[Q[inm, 2] < zf + 0.5 * (np.percentile(Q[inm, 2], 90) - zf) + 0.2]
center = np.median(obj, 0)
floor_y = np.percentile(P[op > 0.4, 1], 97)                     # y is down: the floor is near the largest y
# camera: on the panorama camera's side of the object, `dist` away horizontally, `height` above the floor
side = C0 - center; side[1] = 0; side /= np.linalg.norm(side)
c, s = np.cos(turn), np.sin(turn); side = np.array([c * side[0] + s * side[2], 0.0, -s * side[0] + c * side[2]])
C = center + dist * side; C[1] = floor_y - height
z = center - C; z /= np.linalg.norm(z); x = np.cross([0.0, 1.0, 0.0], z); x /= np.linalg.norm(x)
Rb = np.stack([x, np.cross(z, x), z], 1)

g = lambda *k: torch.tensor(np.stack([v[n] for n in k], 1).astype(np.float32), device="cuda")
qt = g("rot_0", "rot_1", "rot_2", "rot_3")
view = np.eye(4, dtype=np.float32); view[:3, :3] = Rb.T; view[:3, 3] = -Rb.T @ C
K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]], np.float32)
img, alpha, _ = rasterization(g("x", "y", "z"), qt / qt.norm(dim=1, keepdim=True), torch.exp(g("scale_0", "scale_1", "scale_2")),
                              torch.sigmoid(g("opacity")[:, 0]), (0.5 + 0.28209479177387814 * g("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0, 1),
                              torch.tensor(view, device="cuda")[None], torch.tensor(K, device="cuda")[None], S, S, near_plane=0.01)
rgb = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
os.makedirs(f"{out}/frames", exist_ok=True)
cv2.imwrite(f"{out}/crop.jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(f"{out}/frames/0000.jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
json.dump(dict(fov=fov, size=S, f=f, Rb=Rb.tolist(), C=C.tolist(), source=os.path.basename(os.path.normpath(src)),
               object_center=center.tolist(), floor_y=float(floor_y), height=height, dist=dist),
          open(f"{out}/cam.json", "w"), indent=1)
print(f"[novel] object centre {center.round(3)}, floor y {floor_y:.3f}; camera {C.round(3)} "
      f"({height} above floor, {dist} out), alpha mean {alpha.mean().item():.3f} -> {out}/crop.jpg")
