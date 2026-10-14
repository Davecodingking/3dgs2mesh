# Step mask: SAM3 on the view image. The object: text prompt; of the instances found, the one under the click when the
# object has one (the page's "click the object"), else the most confident -> mask_base.png. Then what lies on it (parts:
# pillow, blanket ...; see common.add_parts) -> mask_final.png. Without parts mask_final = mask_base.
# Also writes per instance mask_<prompt>_<id>.png, overlay_<prompt>.jpg and mask_<prompt>.json; overlay_final.jpg
# (object green, added parts orange) and mask_parts.json.
import sys, json, numpy as np, cv2
from .common import sam3, OneImage, add_parts


def tint(img, m, bgr): img[m] = (0.5 * img[m] + 0.5 * np.array(bgr)).astype(np.uint8)


def run(c):
    sam = OneImage(sam3(), f"{c.dir}/frames")
    found = sam.find(c.prompt)
    img = cv2.imread(f"{c.dir}/crop.jpg"); ov = img.copy(); info = []
    tag = c.prompt.replace(" ", "_")
    for oid, m, score in found:
        cv2.imwrite(f"{c.dir}/mask_{tag}_{oid}.png", m.astype(np.uint8) * 255)
        tint(ov, m, (60, 200, 90))
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cs, -1, (0, 255, 255), 2)
        ys, xs = np.nonzero(m)
        cv2.putText(ov, f"#{oid} {score:.2f}", (int(xs.min()), max(30, int(ys.min()) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        info.append(dict(id=oid, score=score, pixels=int(m.sum()), bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]))
    cv2.imwrite(f"{c.dir}/overlay_{tag}.jpg", np.concatenate([img, ov], 1))
    json.dump(info, open(f"{c.dir}/mask_{tag}.json", "w"), indent=1)
    if not found: sys.exit(f"[mask] SAM3 found nothing for '{c.prompt}'")
    if c.click:
        u, v = (int(round(x)) for x in c.click)
        hit = [x for x in found if 0 <= v < x[1].shape[0] and 0 <= u < x[1].shape[1] and x[1][v, u]]
        if not hit: sys.exit(f"[mask] 点的位置没找到物体: none of the {len(found)} '{c.prompt}' instance(s) covers pixel ({u}, {v})")
        best = max(hit, key=lambda x: x[2]); how = f"under the click ({u}, {v})"
    else:
        best = max(found, key=lambda x: x[2]); how = "most confident"
    base = best[1]
    cv2.imwrite(f"{c.dir}/mask_base.png", base.astype(np.uint8) * 255)
    print(f"[mask] '{c.prompt}': {len(info)} instance(s), took #{best[0]} ({how})", info)

    final, added, log = add_parts(sam, base, c.parts) if c.parts else (base, np.zeros_like(base), [])
    sam.close()
    cv2.imwrite(f"{c.dir}/mask_final.png", final.astype(np.uint8) * 255)
    ov = img.copy(); tint(ov, base, (60, 200, 90)); tint(ov, added, (30, 140, 255))
    cs, _ = cv2.findContours(final.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov, cs, -1, (0, 255, 255), 2)
    cv2.imwrite(f"{c.dir}/overlay_final.jpg", np.concatenate([img, ov], 1))
    json.dump(log, open(f"{c.dir}/mask_parts.json", "w"), indent=1)
    for l in log: print("[mask] part", l)
    print(f"[mask] base {int(base.sum())} px + parts {int(added.sum())} px -> mask_final {int(final.sum())} px")
