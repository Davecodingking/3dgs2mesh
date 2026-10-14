# Experiment: look at the replacement. Renders the 3DGS (gsplat) and the placed mesh (pyrender, unlit: its texture is
# baked from the photo) and composites them by depth.
#   rows    crop camera / camera moved 0.3 to the left / 0.3 to the right, both looking at the object
#   columns photo (crop camera only) | 3DGS before | 3DGS after deleting | after deleting + mesh
# Also: mesh silhouette vs SAM3 mask on the crop (IoU), saved as check_silhouette.jpg; numbers in check.json.
# usage: check.py <run_dir> <scene.ply> <carved.ply> <name>
import os, sys, json, numpy as np, torch, cv2
os.environ["PYOPENGL_PLATFORM"] = "egl"
import trimesh, pyrender
from plyfile import PlyData
from gsplat import rasterization
from PIL import Image, ImageDraw, ImageFont

run, scene_ply, carved_ply, name = sys.argv[1:5]
FONT = "/home/ubuntu/pixie_runs/exp_nonerf/code/assets/NotoSansCJK-Regular.otf"
cam = json.load(open(f"{run}/cam.json")); S = int(cam["size"]); f = float(cam["f"])
K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]], np.float32)


def load_gs(path):
    v = PlyData.read(path)["vertex"].data
    g = lambda *k: torch.tensor(np.stack([v[n] for n in k], 1).astype(np.float32), device="cuda")
    q = g("rot_0", "rot_1", "rot_2", "rot_3")
    return dict(means=g("x", "y", "z"), quats=q / q.norm(dim=1, keepdim=True), scales=torch.exp(g("scale_0", "scale_1", "scale_2")),
                opacities=torch.sigmoid(g("opacity")[:, 0]), colors=(0.5 + 0.28209479177387814 * g("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0, 1))
def render_gs(gs, Rcw, C):
    """Rcw: camera axes (x right, y down, z forward) as columns in world; C: camera position."""
    view = np.eye(4, dtype=np.float32); view[:3, :3] = Rcw.T; view[:3, 3] = -Rcw.T @ C
    img, alpha, _ = rasterization(**gs, viewmats=torch.tensor(view, device="cuda")[None], Ks=torch.tensor(K, device="cuda")[None],
                                  width=S, height=S, near_plane=0.01, render_mode="RGB+ED")
    rgb = (img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return rgb, img[0, ..., 3].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()
renderer = pyrender.OffscreenRenderer(S, S)
meshes = trimesh.load(f"{run}/gen/{name}_world.glb", force="scene").dump()
def render_mesh(Rcw, C):
    sc = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0])
    for m in meshes: sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    pose = np.eye(4); pose[:3, :3] = Rcw @ np.diag([1.0, -1.0, -1.0]); pose[:3, 3] = C      # OpenCV -> OpenGL camera
    sc.add(pyrender.IntrinsicsCamera(fx=f, fy=f, cx=S / 2, cy=S / 2, znear=0.01, zfar=100.0), pose=pose)
    col, dep = renderer.render(sc, flags=pyrender.RenderFlags.FLAT)
    return col[..., :3], dep
def look_at(C, target):
    z = target - C; z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z); x /= np.linalg.norm(x)       # world y is down (OpenCV), so this is "right"
    return np.stack([x, np.cross(z, x), z], 1)


gs_before, gs_after, gs_removed = load_gs(scene_ply), load_gs(carved_ply), load_gs(f"{run}/removed.ply")
V = np.concatenate([m.vertices for m in meshes]); center = (V.min(0) + V.max(0)) / 2
Rb = np.array(cam["Rb"]); C0 = np.array(cam.get("C", [0.0, 0.0, 0.0]))
side = Rb[:, 0] * np.array([1.0, 0.0, 1.0]); side /= np.linalg.norm(side)
views = [("切图相机", Rb, C0)]
for label, sgn in (("相机左移 0.3", -1), ("相机右移 0.3", 1)):
    C = C0 + sgn * 0.3 * side; views.append((label, look_at(C, center), C))

T = 512; font = ImageFont.truetype(FONT, 22); rows = []; nums = {}
for vi, (label, Rcw, C) in enumerate(views):
    rb, _, _ = render_gs(gs_before, Rcw, C)
    ra, da, aa = render_gs(gs_after, Rcw, C)
    mc, md = render_mesh(Rcw, C)
    hit = md > 0
    on_top = hit & ((md < da) | (aa < 0.5))
    comp = ra.copy(); comp[on_top] = mc[on_top]
    rr, _, _ = render_gs(gs_removed, Rcw, C)                 # what was deleted: anything besides the object is collateral
    first = cv2.imread(f"{run}/crop.jpg")[..., ::-1] if vi == 0 else np.full((S, S, 3), 255, np.uint8)
    tiles = [first, rb, ra, comp, rr]
    tiles = [Image.fromarray(np.ascontiguousarray(t)).resize((T, T)) for t in tiles]
    heads = ["原图" if vi == 0 else "", "原来的 3DGS", "删点后", "删点后 + 网格", "删掉的点"]
    for t, hd in zip(tiles, heads):
        d = ImageDraw.Draw(t)
        if hd: d.rectangle([0, 0, 24 * len(hd) + 16, 34], fill=(0, 0, 0)); d.text((8, 4), hd, font=font, fill=(255, 255, 255))
    d = ImageDraw.Draw(tiles[0]) if vi else None
    if d: d.text((20, T // 2 - 14), label, font=font, fill=(0, 0, 0))
    elif vi == 0: ImageDraw.Draw(tiles[0]).text((8, T - 34), label, font=font, fill=(255, 255, 0))
    rows.append(np.concatenate([np.asarray(t) for t in tiles], 1))
    if vi == 0:                                              # silhouette vs mask on the crop camera
        mask = np.array(Image.open(f"{run}/mask_final.png").convert("L")) > 127
        iou = float((hit & mask).sum() / max(1, (hit | mask).sum()))
        nums.update(silhouette_iou=iou, mesh_px=int(hit.sum()), mask_px=int(mask.sum()))
        ov = cv2.imread(f"{run}/crop.jpg")
        for m_, colr in ((mask, (0, 255, 255)), (hit, (0, 0, 255))):
            cs, _ = cv2.findContours(m_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cs, -1, colr, 3)
        cv2.imwrite(f"{run}/check_silhouette.jpg", ov)
Image.fromarray(np.concatenate(rows, 0)).save(f"{run}/check.jpg", quality=90)
json.dump(nums, open(f"{run}/check.json", "w"), indent=1)
print(f"[check] silhouette IoU {nums['silhouette_iou']:.3f} (mesh {nums['mesh_px']} px, mask {nums['mask_px']} px) "
      f"-> {run}/check.jpg, {run}/check_silhouette.jpg")
