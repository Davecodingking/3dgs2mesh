# Experiment: cut a square perspective crop out of an equirectangular panorama.
# Camera convention copied from 3dgs2mesh/place.mjs (dirv / basis): OpenCV world (x right, y down, z forward),
# panorama camera at the origin, yaw 0 = panorama centre (positive to the right), pitch positive up.
# usage: crop.py <pano.jpg> <out_dir> <yaw> <pitch> <fov> <size>
import sys, os, json, numpy as np, cv2

pano, out = sys.argv[1], sys.argv[2]
yaw, pitch, fov, size = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])

y, p = np.radians(yaw), np.radians(pitch)
z = np.array([np.sin(y) * np.cos(p), -np.sin(p), np.cos(y) * np.cos(p)])
x = np.cross([0.0, 1.0, 0.0], z); x /= np.linalg.norm(x)
Rb = np.stack([x, np.cross(z, x), z], 1)             # columns: camera right / down / forward in world
f = size / 2 / np.tan(np.radians(fov) / 2)

j, i = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
rays = np.stack([(j - size / 2) / f, (i - size / 2) / f, np.ones_like(j)], -1) @ Rb.T
rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
lon, lat = np.arctan2(rays[..., 0], rays[..., 2]), np.arcsin(-rays[..., 1])

img = cv2.imread(pano); H, W = img.shape[:2]
u = ((lon / (2 * np.pi) + 0.5) * W - 0.5).astype(np.float32)
v = ((0.5 - lat / np.pi) * H - 0.5).astype(np.float32)
crop = cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

os.makedirs(f"{out}/frames", exist_ok=True)
cv2.imwrite(f"{out}/crop.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(f"{out}/frames/0000.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
json.dump(dict(yaw=yaw, pitch=pitch, fov=fov, size=size, f=f, Rb=Rb.tolist()), open(f"{out}/cam.json", "w"), indent=1)
print(f"[crop] yaw {yaw} pitch {pitch} fov {fov} -> {out}/crop.jpg")
