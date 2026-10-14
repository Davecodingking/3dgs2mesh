# Experiment: add the things lying on the object (pillow, blanket) to its mask.
# Each extra word is run through SAM3 on the same crop; an instance is kept when it touches the base mask
# and at least half of it lies inside the base mask's box. Union + small close -> mask_final.png.
# usage: mask_parts.py <out_dir> <base_mask.png> <word> [<word> ...]
import sys, os, json, numpy as np, cv2
from sam3.model_builder import build_sam3_video_predictor

out, base_png, words = sys.argv[1], sys.argv[2], sys.argv[3:]
base = cv2.imread(f"{out}/{base_png}", 0) > 127
ys, xs = np.nonzero(base); x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
pad = 0.05 * max(x1 - x0, y1 - y0)
inbox = np.zeros_like(base); inbox[int(max(0, y0 - pad)):int(y1 + pad) + 1, int(max(0, x0 - pad)):int(x1 + pad) + 1] = True
near = cv2.dilate(base.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0

pred = build_sam3_video_predictor(checkpoint_path=os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt"))
sid = pred.handle_request(request=dict(type="start_session", resource_path=f"{out}/frames"))["session_id"]
union, added, log = base.copy(), np.zeros_like(base), []
for w in words:
    pred.handle_request(request=dict(type="reset_session", session_id=sid))
    pred.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text=w))
    res = None
    for r in pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
        if r["frame_index"] == 0: res = r["outputs"]
    for k, oid in enumerate(res["out_obj_ids"].tolist()):
        m = res["out_binary_masks"][k].astype(bool)
        if not m.any(): continue
        frac_box, touches = float((m & inbox).sum() / m.sum()), bool((m & near).any())
        keep = touches and frac_box >= 0.5
        log.append(dict(word=w, id=oid, score=round(float(res["out_probs"][k]), 3), pixels=int(m.sum()),
                        inside_box=round(frac_box, 2), touches=touches, kept=keep))
        if keep: added |= m & ~base; union |= m

final = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
cv2.imwrite(f"{out}/mask_final.png", final.astype(np.uint8) * 255)
img = cv2.imread(f"{out}/crop.jpg"); ov = img.copy()
ov[base] = (0.5 * ov[base] + 0.5 * np.array([60, 200, 90])).astype(np.uint8)       # object: green
ov[added] = (0.5 * ov[added] + 0.5 * np.array([30, 140, 255])).astype(np.uint8)    # added parts: orange
cs, _ = cv2.findContours(final.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(ov, cs, -1, (0, 255, 255), 2)
cv2.imwrite(f"{out}/overlay_final.jpg", np.concatenate([img, ov], 1))
json.dump(log, open(f"{out}/mask_parts.json", "w"), indent=1)
for l in log: print("[parts]", l)
print(f"[parts] base {int(base.sum())} px + added {int(added.sum())} px -> final {int(final.sum())} px")
