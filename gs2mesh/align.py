# Step align: the 3DGS rendered from the view camera next to the photo crop, to check that panorama and splats line up.
# Writes gs_render.jpg, gs_depth.npy and check_align.jpg (crop | render | blend). A rendered view is the 3DGS already: skipped.
import numpy as np, cv2
from .common import load_cam, read_ply, gs_tensors, render


def run(c):
    cam = load_cam(c.dir)
    if cam["C"].any():
        print("[align] the view is a 3DGS render, nothing to check"); return
    v = read_ply(c.ply)
    rgb, depth, alpha = render(gs_tensors(v), cam["Rb"], cam["C"], cam["f"], cam["size"]); rgb = rgb[..., ::-1]
    np.save(f"{c.dir}/gs_depth.npy", depth)
    cv2.imwrite(f"{c.dir}/gs_render.jpg", rgb)
    crop = cv2.imread(f"{c.dir}/crop.jpg")
    cv2.imwrite(f"{c.dir}/check_align.jpg", np.concatenate([crop, rgb, ((crop.astype(int) + rgb) // 2).astype(np.uint8)], 1))
    print(f"[align] {len(v)} gaussians, alpha mean {alpha.mean():.3f} -> {c.dir}/check_align.jpg")
