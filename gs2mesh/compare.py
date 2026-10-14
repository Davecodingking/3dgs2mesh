# Step compare: per-view comparison images for the check page. For every view with a mask, renders from that view's camera
#   cmp_before.jpg   the original 3DGS
#   cmp_after.jpg    the 3DGS after deletion; pixels inside the mask that still show the same surface as before (depth
#                    within 3%) are tinted magenta: that is what the mask says should go and did not
#   cmp_removed.jpg  only the deleted gaussians, on black
# all with the view's mask outline in yellow. Writes views/<k>/cmp_*.jpg and compare.json (leftover share per view).
import json, numpy as np, cv2
from .common import read_ply, gs_tensors, render

OUT = 768


def run(c):
    meta = json.load(open(f"{c.dir}/views.json"))
    before, after, removed = (gs_tensors(read_ply(p)) for p in (c.ply, f"{c.dir}/scene_carved.ply", f"{c.dir}/removed.ply"))
    report = {}
    for vw in meta["views"]:
        if not vw.get("mask"): continue
        d = f"{c.dir}/views/{vw['k']}"; cam = (vw["Rb"], vw["C"], vw["f"], vw["size"])
        M = cv2.imread(f"{d}/mask.png", 0) > 127
        rb, db, ab = render(before, *cam); ra, da, aa = render(after, *cam); rr, _, ar = render(removed, *cam)
        same = M & (ab > 0.5) & (aa > 0.5) & (np.abs(da - db) < 0.03 * db)            # still the old surface inside the mask
        ra = ra.copy(); ra[same] = (0.45 * ra[same] + 0.55 * np.array([255, 40, 220])).astype(np.uint8)
        rr = (rr * ar[..., None]).astype(np.uint8)
        cs, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for name, img in (("before", rb), ("after", ra), ("removed", rr)):
            im = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR); cv2.drawContours(im, cs, -1, (0, 255, 255), 3)
            cv2.imwrite(f"{d}/cmp_{name}.jpg", cv2.resize(im, (OUT, OUT), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
        report[vw["k"]] = dict(name=vw["name"], leftover=round(float(same.sum() / max(1, M.sum())), 4), mask_px=int(M.sum()))
        print(f"[compare] view {vw['k']} ({vw['name']}): inside the mask still the old surface {100 * report[vw['k']]['leftover']:.1f}%", flush=True)
    json.dump(report, open(f"{c.dir}/compare.json", "w"), indent=1)
