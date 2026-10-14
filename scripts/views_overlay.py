# Experiment: every masked view with the SAM3 mask outline (yellow) and the placed mesh's real outline (red, rendered with
# pyrender from that view's camera) -> <run>/views_overlay.jpg. For judging the multi-view fit by eye.
# usage: views_overlay.py <run> <name> [glb=gen/<name>_world.glb]
import os, sys, json, numpy as np, cv2
os.environ["PYOPENGL_PLATFORM"] = "egl"
import trimesh, pyrender

run, name = sys.argv[1:3]
glb = sys.argv[3] if len(sys.argv) > 3 else f"{run}/gen/{name}_world.glb"
meta = json.load(open(f"{run}/views.json"))
meshes = trimesh.load(glb, force="scene").dump()
tiles = []; r = None
for vw in meta["views"]:
    if not vw.get("mask"): continue
    d = f"{run}/views/{vw['k']}"; S = vw["size"]; f = vw["f"]
    if r is None: r = pyrender.OffscreenRenderer(S, S)
    sc = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
    for m in meshes: sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    pose = np.eye(4); pose[:3, :3] = np.array(vw["Rb"]) @ np.diag([1.0, -1.0, -1.0]); pose[:3, 3] = vw["C"]
    sc.add(pyrender.IntrinsicsCamera(fx=f, fy=f, cx=S / 2, cy=S / 2, znear=0.01, zfar=100.0), pose=pose)
    _, dep = r.render(sc, flags=pyrender.RenderFlags.FLAT)
    img = cv2.imread(f"{d}/frames/0000.jpg"); M = cv2.imread(f"{d}/mask.png", 0) > 127; sil = dep > 0
    for m_, col in ((M, (0, 255, 255)), (sil, (0, 0, 255))):
        cs, _ = cv2.findContours(m_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(img, cs, -1, col, 3)
    iou = (sil & M).sum() / max(1, (sil | M).sum())
    cv2.putText(img, f"{vw['name']}  IoU {iou:.2f}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    tiles.append(cv2.resize(img, (512, 512)))
while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(f"{run}/views_overlay.jpg", np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)], 0),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"[views_overlay] {len(tiles)} tiles -> {run}/views_overlay.jpg")
