# Shared pieces of the steps: reading / writing / rendering the 3DGS, cameras, SAM3 on one image, the .splat format.
# World frame everywhere: OpenCV (x right, y down, z forward), the panorama camera at the origin, yaw 0 = panorama centre.
# A camera is a dict {size, f, Rb, C}: square image, focal length in px, Rb = camera axes (x right, y down, z forward) as
# columns in world, C = position; K = [[f, 0, S/2], [0, f, S/2], [0, 0, 1]].
# Heavy imports (torch, gsplat, sam3) stay inside the functions: the SAM3 env has no gsplat, the SAM 3D env no sam3.
import os, json, numpy as np

SH0 = 0.28209479177387814
SAM3_CKPT = os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt")
SAM3D_DIR = "/home/ubuntu/sam3d/sam-3d-objects"
FONTS = [os.environ.get("G2M_FONT", ""), "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/home/ubuntu/pixie_runs/exp_nonerf/code/assets/NotoSansCJK-Regular.otf"]


def sigmoid(x): return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- 3DGS files
def read_ply(path):
    from plyfile import PlyData
    return PlyData.read(path)["vertex"].data


def write_ply(v, path):
    from plyfile import PlyData, PlyElement
    PlyData([PlyElement.describe(v, "vertex")]).write(path)


def xyz(v): return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
def opacity(v): return sigmoid(v["opacity"].astype(np.float64))


def to_splat(v, path, min_alpha=0.02):
    """3DGS vertex array -> the 32-byte .splat format the web viewer loads (gaussian-splats-3d): position f32x3, linear
    scale f32x3, RGBA u8x4 (SH0 colour, sigmoid alpha), rotation u8x4 (w, x, y, z). Faint ones dropped. Returns the count."""
    a = opacity(v); keep = a >= min_alpha; v, a = v[keep], a[keep]
    buf = np.zeros(len(v), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    buf["p"] = np.stack([v["x"], v["y"], v["z"]], 1)
    buf["s"] = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    rgb = 0.5 + SH0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    buf["c"] = np.clip(np.round(np.c_[rgb, a] * 255), 0, 255).astype(np.uint8)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    buf["r"] = np.clip(np.round(q / np.linalg.norm(q, axis=1, keepdims=True) * 128 + 128), 0, 255).astype(np.uint8)
    buf.tofile(path); return len(buf)


# ---------------------------------------------------------------- rendering the 3DGS (gsplat, DC colour only)
def gs_tensors(v):
    """Vertex array -> the keyword arguments gsplat's rasterization takes for the gaussians."""
    import torch
    g = lambda *k: torch.tensor(np.stack([v[n] for n in k], 1).astype(np.float32), device="cuda")
    q = g("rot_0", "rot_1", "rot_2", "rot_3")
    return dict(means=g("x", "y", "z"), quats=q / q.norm(dim=1, keepdim=True), scales=torch.exp(g("scale_0", "scale_1", "scale_2")),
                opacities=torch.sigmoid(g("opacity")[:, 0]), colors=(0.5 + SH0 * g("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0, 1))


def render(gs, Rb, C, f, S):
    """(rgb uint8 S x S x 3 in RGB order, expected depth S x S, alpha S x S) seen by the camera Rb / C / f, square S px."""
    import torch
    from gsplat import rasterization
    Rb, C = np.asarray(Rb, float), np.asarray(C, float)
    view = np.eye(4, dtype=np.float32); view[:3, :3] = Rb.T; view[:3, 3] = -Rb.T @ C
    K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]], np.float32)
    img, alpha, _ = rasterization(**gs, viewmats=torch.tensor(view, device="cuda")[None], Ks=torch.tensor(K, device="cuda")[None],
                                  width=S, height=S, near_plane=0.01, render_mode="RGB+ED")
    return ((img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), img[0, ..., 3].cpu().numpy(),
            alpha[0, ..., 0].cpu().numpy())


# ---------------------------------------------------------------- cameras
def load_cam(run_dir):
    cam = json.load(open(f"{run_dir}/cam.json"))
    cam["Rb"] = np.array(cam["Rb"], float); cam["C"] = np.array(cam.get("C", [0.0, 0.0, 0.0]), float)
    return cam


def look_at(C, target):
    """Camera axes (columns) for a camera at C looking at target, level (no roll). World y is down, so x is 'right'."""
    z = np.asarray(target, float) - np.asarray(C, float); z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def yaw_pitch(yaw, pitch):
    """Camera axes for the panorama camera turned yaw deg right and pitch deg up (the convention of place.mjs)."""
    y, p = np.radians(yaw), np.radians(pitch)
    z = np.array([np.sin(y) * np.cos(p), -np.sin(p), np.cos(y) * np.cos(p)])
    x = np.cross([0.0, 1.0, 0.0], z); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def focal(fov_deg, S): return S / 2 / np.tan(np.radians(fov_deg) / 2)


def pano_crop(pano_path, Rb, f, S):
    """Square perspective image (BGR) seen from the panorama centre by camera axes Rb, sampled from the equirectangular photo."""
    import cv2
    j, i = np.meshgrid(np.arange(S) + 0.5, np.arange(S) + 0.5)
    rays = np.stack([(j - S / 2) / f, (i - S / 2) / f, np.ones_like(j)], -1) @ np.asarray(Rb, float).T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    lon, lat = np.arctan2(rays[..., 0], rays[..., 2]), np.arcsin(-rays[..., 1])
    img = cv2.imread(pano_path); H, W = img.shape[:2]
    u = ((lon / (2 * np.pi) + 0.5) * W - 0.5).astype(np.float32)
    v = ((0.5 - lat / np.pi) * H - 0.5).astype(np.float32)
    return cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# ---------------------------------------------------------------- SAM3 on one image
def sam3():
    from sam3.model_builder import build_sam3_video_predictor
    return build_sam3_video_predictor(checkpoint_path=SAM3_CKPT)


class OneImage:
    """SAM3's video predictor on a one-frame "video" (a folder holding 0000.jpg), as in
    sam3d/data/outdoor/scripts/sam3_video_probe.py. find(word) -> [(object id, bool mask, score)], empty masks dropped."""

    def __init__(self, pred, frames_dir):
        self.pred, self.used = pred, False
        self.sid = pred.handle_request(request=dict(type="start_session", resource_path=frames_dir))["session_id"]

    def find(self, word):
        if self.used: self.pred.handle_request(request=dict(type="reset_session", session_id=self.sid))
        self.used = True
        self.pred.handle_request(request=dict(type="add_prompt", session_id=self.sid, frame_index=0, text=word))
        res = None
        for r in self.pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=self.sid)):
            if r["frame_index"] == 0: res = r["outputs"]
        out = []
        for k, oid in enumerate(res["out_obj_ids"].tolist()):
            m = res["out_binary_masks"][k].astype(bool)
            if m.any(): out.append((oid, m, float(res["out_probs"][k])))
        return out

    def close(self): self.pred.handle_request(request=dict(type="close_session", session_id=self.sid))


def add_parts(img, base, words):
    """What lies on the object (pillow, blanket): an instance of a part word joins the mask when it touches the object's
    mask and at least half of it lies inside the mask's box (+5%). Returns (final mask, added pixels, log)."""
    import cv2
    ys, xs = np.nonzero(base); x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    pad = 0.05 * max(x1 - x0, y1 - y0)
    box = np.zeros_like(base); box[int(max(0, y0 - pad)):int(y1 + pad) + 1, int(max(0, x0 - pad)):int(x1 + pad) + 1] = True
    near = cv2.dilate(base.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    union, added, log = base.copy(), np.zeros_like(base), []
    for w in words:
        for oid, m, score in img.find(w):
            inside, touches = float((m & box).sum() / m.sum()), bool((m & near).any())
            keep = touches and inside >= 0.5
            log.append(dict(word=w, id=oid, score=round(score, 3), pixels=int(m.sum()), inside_box=round(inside, 2),
                            touches=touches, kept=keep))
            if keep: added |= m & ~base; union |= m
    final = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    return final, added, log


def font(size):
    """A CJK font for labels on the check images, or None (labels then fall back to English)."""
    from PIL import ImageFont
    for p in FONTS:
        if p and os.path.exists(p): return ImageFont.truetype(p, size)
    return None
