# Experiment: per-view comparison images for the check page. For every view with a mask, renders from that view's camera
#   cmp_before.jpg   the original 3DGS
#   cmp_after.jpg    the 3DGS after deletion; pixels inside the mask that still show the same surface as before (depth
#                    within 3%) are tinted magenta: that is what the mask says should go and did not
#   cmp_removed.jpg  only the deleted gaussians, on black
# all with the view's mask outline in yellow. Writes views/<k>/cmp_*.jpg and compare.json (leftover share per view).
# usage: view_compare.py <run> <scene.ply> <carved.ply>
import sys, json, numpy as np, torch, cv2
from plyfile import PlyData
from gsplat import rasterization

run, scene_ply, carved_ply = sys.argv[1:4]
OUT = 768
meta = json.load(open(f"{run}/views.json"))


def load(path):
    v = PlyData.read(path)["vertex"].data
    g = lambda *k: torch.tensor(np.stack([v[n] for n in k], 1).astype(np.float32), device="cuda")
    q = g("rot_0", "rot_1", "rot_2", "rot_3")
    return dict(means=g("x", "y", "z"), quats=q / q.norm(dim=1, keepdim=True), scales=torch.exp(g("scale_0", "scale_1", "scale_2")),
                opacities=torch.sigmoid(g("opacity")[:, 0]), colors=(0.5 + 0.28209479177387814 * g("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0, 1))


def render(gs, vw):
    S, f = vw["size"], vw["f"]; Rb, C = np.array(vw["Rb"]), np.array(vw["C"])
    view = np.eye(4, dtype=np.float32); view[:3, :3] = Rb.T; view[:3, 3] = -Rb.T @ C
    K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]], np.float32)
    img, alpha, _ = rasterization(**gs, viewmats=torch.tensor(view, device="cuda")[None], Ks=torch.tensor(K, device="cuda")[None],
                                  width=S, height=S, near_plane=0.01, render_mode="RGB+ED")
    return (img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), img[0, ..., 3].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()


before, after, removed = load(scene_ply), load(carved_ply), load(f"{run}/removed.ply")
report = {}
for vw in meta["views"]:
    if not vw.get("mask"): continue
    d = f"{run}/views/{vw['k']}"
    M = cv2.imread(f"{d}/mask.png", 0) > 127
    rb, db, ab = render(before, vw); ra, da, aa = render(after, vw); rr, _, ar = render(removed, vw)
    same = M & (ab > 0.5) & (aa > 0.5) & (np.abs(da - db) < 0.03 * db)            # still the old surface inside the mask
    ra = ra.copy(); ra[same] = (0.45 * ra[same] + 0.55 * np.array([255, 40, 220])).astype(np.uint8)
    rr = (rr * ar[..., None]).astype(np.uint8)
    cs, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for name, img in (("before", rb), ("after", ra), ("removed", rr)):
        im = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR); cv2.drawContours(im, cs, -1, (0, 255, 255), 3)
        cv2.imwrite(f"{d}/cmp_{name}.jpg", cv2.resize(im, (OUT, OUT), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    report[vw["k"]] = dict(name=vw["name"], leftover=round(float(same.sum() / max(1, M.sum())), 4), mask_px=int(M.sum()))
    print(f"[view_compare] view {vw['k']} ({vw['name']}): inside the mask still the old surface {100 * report[vw['k']]['leftover']:.1f}%", flush=True)
json.dump(report, open(f"{run}/compare.json", "w"), indent=1)
