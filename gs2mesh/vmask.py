# Step vmask: SAM3 text prompt on every view from the views step. SAM3 may find several instances (or the wrong one) in a
# view; the one kept covers most of the object's seed points projected into that view. A view where no instance covers at
# least 30% of the seed pixels gets no mask and is left out of the vote. Parts (pillow, blanket ...) join as in the mask step.
# Given views come with their mask. Writes views/<k>/mask.png, views/<k>/overlay.jpg and the per-view results into views.json.
import os, json, numpy as np, cv2
from .common import sam3, OneImage, add_parts


def run(c):
    meta = json.load(open(f"{c.dir}/views.json")); pred = sam3()
    for vw in meta["views"]:
        if vw.get("given"):
            print(f"[vmask] view {vw['k']} ({vw['name']}): given mask", flush=True); continue
        d = f"{c.dir}/views/{vw['k']}"
        img_s = OneImage(pred, f"{d}/frames"); found = img_s.find(c.prompt)
        img = cv2.imread(f"{d}/frames/0000.jpg"); S = img.shape[0]
        sp = np.load(f"{d}/seed_px.npy"); sp = sp[(sp[:, 0] >= 0) & (sp[:, 0] < S) & (sp[:, 1] >= 0) & (sp[:, 1] < S)].astype(np.int64)
        cover = lambda m: float(m[sp[:, 1], sp[:, 0]].mean()) if len(sp) else 0.0
        best, best_cov = None, 0.0
        for _, m, score in found:
            if cover(m) > best_cov: best, best_cov = (m, score), cover(m)
        vw["instances"] = len(found); vw["seed_cover"] = round(best_cov, 3)
        if best is not None and best_cov >= 0.3 and c.parts:
            best = (add_parts(img_s, best[0], c.parts)[0], best[1]); best_cov = cover(best[0])
            vw["seed_cover"] = round(best_cov, 3)
        img_s.close()
        ov = img.copy()
        if best is not None and best_cov >= 0.3:
            m, score = best; vw["mask"] = True; vw["score"] = round(score, 3); vw["mask_px"] = int(m.sum())
            cv2.imwrite(f"{d}/mask.png", m.astype(np.uint8) * 255)
            ov[m] = (0.5 * ov[m] + 0.5 * np.array([60, 200, 90])).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cs, -1, (0, 255, 255), 2)
        else:
            vw["mask"] = False
            if os.path.exists(f"{d}/mask.png"): os.remove(f"{d}/mask.png")
        for x, y in sp[:: max(1, len(sp) // 400)]: cv2.circle(ov, (int(x), int(y)), 2, (255, 0, 255), -1)   # seed points
        cv2.imwrite(f"{d}/overlay.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[vmask] view {vw['k']} ({vw['name']}): {vw['instances']} instance(s), seed cover {best_cov:.2f}"
              + (f", score {vw['score']}" if vw.get("mask") else ", NO MASK"), flush=True)
    json.dump(meta, open(f"{c.dir}/views.json", "w"), indent=1)
