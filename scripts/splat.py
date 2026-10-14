# Experiment: 3DGS vertex array -> the 32-byte .splat format the web viewers load (gaussian-splats-3d):
# position f32x3, linear scale f32x3, RGBA u8x4 (SH0 colour, sigmoid alpha), rotation u8x4 (w,x,y,z). Faint ones dropped.
import numpy as np


def to_splat(v, path, min_alpha=0.02):
    a = 1 / (1 + np.exp(-v["opacity"].astype(np.float64))); keep = a >= min_alpha; v, a = v[keep], a[keep]
    buf = np.zeros(len(v), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    buf["p"] = np.stack([v["x"], v["y"], v["z"]], 1)
    buf["s"] = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    rgb = 0.5 + 0.28209479177387814 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    buf["c"] = np.clip(np.round(np.c_[rgb, a] * 255), 0, 255).astype(np.uint8)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    buf["r"] = np.clip(np.round(q / np.linalg.norm(q, axis=1, keepdims=True) * 128 + 128), 0, 255).astype(np.uint8)
    buf.tofile(path); return len(buf)
