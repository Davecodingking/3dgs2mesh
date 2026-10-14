# Experiment: render the 3DGS scene from the crop camera (camera at origin) to check the
# panorama and the splats line up. Saves gs_render.jpg, gs_depth.npy and a side-by-side check image.
# usage: render_gs.py <scene.ply> <out_dir>
import sys, json, numpy as np, torch, cv2
from plyfile import PlyData
from gsplat import rasterization

ply, out = sys.argv[1], sys.argv[2]
cam = json.load(open(f"{out}/cam.json"))
v = PlyData.read(ply)["vertex"].data
g = lambda *k: torch.tensor(np.stack([v[n] for n in k], 1).astype(np.float32)).cuda()
means = g("x", "y", "z")
quats = g("rot_0", "rot_1", "rot_2", "rot_3"); quats = quats / quats.norm(dim=1, keepdim=True)
scales = torch.exp(g("scale_0", "scale_1", "scale_2"))
opac = torch.sigmoid(g("opacity")[:, 0])
cols = (0.5 + 0.28209479177387814 * g("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0, 1)

S, f = cam["size"], cam["f"]
view = np.eye(4, dtype=np.float32); view[:3, :3] = np.array(cam["Rb"], np.float32).T   # world -> camera
K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]], np.float32)
img, alpha, _ = rasterization(means, quats, scales, opac, cols, torch.tensor(view)[None].cuda(),
                              torch.tensor(K)[None].cuda(), S, S, near_plane=0.01, render_mode="RGB+ED")
rgb = (img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
np.save(f"{out}/gs_depth.npy", img[0, ..., 3].cpu().numpy())
cv2.imwrite(f"{out}/gs_render.jpg", rgb)
crop = cv2.imread(f"{out}/crop.jpg")
cv2.imwrite(f"{out}/check_align.jpg", np.concatenate([crop, rgb, ((crop.astype(int) + rgb) // 2).astype(np.uint8)], 1))
print(f"[render] {len(v)} gaussians, alpha mean {alpha.mean().item():.3f} -> {out}/check_align.jpg")
