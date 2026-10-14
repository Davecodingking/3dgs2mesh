# Experiment: SAM3 text prompt on the single crop (a one-frame "video", same API as
# sam3d/data/outdoor/scripts/sam3_video_probe.py). Saves one mask png per instance and an overlay.
# usage: mask.py <out_dir> <text prompt>
import sys, os, json, numpy as np, cv2, torch
from sam3.model_builder import build_sam3_video_predictor

out, word = sys.argv[1], sys.argv[2]
pred = build_sam3_video_predictor(checkpoint_path=os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt"))
sid = pred.handle_request(request=dict(type="start_session", resource_path=f"{out}/frames"))["session_id"]
pred.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text=word))
res = None
for r in pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
    if r["frame_index"] == 0: res = r["outputs"]
print("[mask] output keys:", list(res.keys()))

img = cv2.imread(f"{out}/crop.jpg"); ov = img.copy(); info = []
tag = word.replace(" ", "_")
for k, oid in enumerate(res["out_obj_ids"].tolist()):
    m = res["out_binary_masks"][k].astype(bool)
    if not m.any(): continue
    score = float(res["out_probs"][k]) if "out_probs" in res else None
    cv2.imwrite(f"{out}/mask_{tag}_{oid}.png", m.astype(np.uint8) * 255)
    ov[m] = (0.5 * ov[m] + 0.5 * np.array([60, 200, 90])).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov, cs, -1, (0, 255, 255), 2)
    ys, xs = np.nonzero(m)
    cv2.putText(ov, f"#{oid}" + (f" {score:.2f}" if score is not None else ""), (int(xs.min()), max(30, int(ys.min()) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    info.append(dict(id=oid, score=score, pixels=int(m.sum()), bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]))
cv2.imwrite(f"{out}/overlay_{tag}.jpg", np.concatenate([img, ov], 1))
json.dump(info, open(f"{out}/mask_{tag}.json", "w"), indent=1)
if not info: sys.exit(f"[mask] SAM3 found nothing for '{word}'")
best = max(info, key=lambda d: d["score"] or 0)                 # the next steps take the most confident instance
cv2.imwrite(f"{out}/mask_base.png", cv2.imread(f"{out}/mask_{tag}_{best['id']}.png", 0))
print(f"[mask] '{word}': {len(info)} instance(s)", info)
