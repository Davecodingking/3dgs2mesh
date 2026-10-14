# Step view: the image SAM 3D generates from, seen by the object's camera (cam: size, f, Rb, C).
# Camera at the panorama centre (C = 0): sampled from the panorama photo. Anywhere else: rendered from the 3DGS (gsplat),
# e.g. lower than the panorama camera when the panorama sees the object too steeply (A01 coffee table: from the panorama
# centre SAM 3D only made a flat disc without its base, 4 seeds out of 4).
# Writes crop.jpg, frames/0000.jpg (SAM3 reads a folder) and cam.json (+ fov).
import os, json, numpy as np, cv2
from .common import pano_crop, read_ply, gs_tensors, render


def run(c):
    S, f = int(c.cam["size"]), float(c.cam["f"])
    Rb, C = np.array(c.cam["Rb"], float), np.array(c.cam.get("C", [0.0, 0.0, 0.0]), float)
    if not C.any():
        img, src = pano_crop(c.pano, Rb, f, S), "panorama"
    else:
        rgb, _, alpha = render(gs_tensors(read_ply(c.ply)), Rb, C, f, S); img, src = rgb[..., ::-1], "3DGS render"
    os.makedirs(f"{c.dir}/frames", exist_ok=True)
    for p in ("crop.jpg", "frames/0000.jpg"): cv2.imwrite(f"{c.dir}/{p}", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    fov = float(np.degrees(2 * np.arctan(S / 2 / f)))
    json.dump(dict(fov=fov, size=S, f=f, Rb=Rb.tolist(), C=C.tolist()), open(f"{c.dir}/cam.json", "w"), indent=1)
    print(f"[view] {src}, camera at {C.round(3).tolist()}, fov {fov:.1f} -> {c.dir}/crop.jpg")
