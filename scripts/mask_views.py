# Experiment: SAM3 text prompt on every view from render_views.py (step 2 of the multi-view flow). SAM3 may find several
# instances (or the wrong one) in a view; the one kept covers most of the object's seed points projected into that view.
# A view where no instance covers at least 30% of the seed pixels gets no mask and is left out of the vote.
# Writes views/<k>/mask.png, views/<k>/overlay.jpg, and the per-view results into views.json.
# Extra words after the prompt (pillow, blanket ...) add what lies on the object, as mask_parts.py does on the crop:
# an instance of an extra word joins when it touches the object's mask and half of it is inside that mask's box.
# usage: mask_views.py <run> "<prompt>" [part word ...]
import os, sys, json, numpy as np, cv2
from sam3.model_builder import build_sam3_video_predictor

run, word, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
meta = json.load(open(f"{run}/views.json"))
pred = build_sam3_video_predictor(checkpoint_path=os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt"))
for vw in meta["views"]:
    if vw.get("given"):                                       # mask came with the view
        print(f"[mask_views] view {vw['k']} ({vw['name']}): given mask", flush=True); continue
    d = f"{run}/views/{vw['k']}"
    sid = pred.handle_request(request=dict(type="start_session", resource_path=f"{d}/frames"))["session_id"]
    pred.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text=word))
    res = None
    for r in pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
        if r["frame_index"] == 0: res = r["outputs"]
    img = cv2.imread(f"{d}/frames/0000.jpg"); S = img.shape[0]
    sp = np.load(f"{d}/seed_px.npy"); sp = sp[(sp[:, 0] >= 0) & (sp[:, 0] < S) & (sp[:, 1] >= 0) & (sp[:, 1] < S)].astype(np.int64)
    best, best_cov = None, 0.0
    for k, oid in enumerate(res["out_obj_ids"].tolist()):
        m = res["out_binary_masks"][k].astype(bool)
        cov = float(m[sp[:, 1], sp[:, 0]].mean()) if len(sp) else 0.0
        if cov > best_cov: best, best_cov = (m, float(res["out_probs"][k])), cov
    vw["instances"] = len(res["out_obj_ids"]); vw["seed_cover"] = round(best_cov, 3)
    ov = img.copy()
    if best is not None and best_cov >= 0.3 and parts:
        m0 = best[0]; ys, xs = np.nonzero(m0); pad = 0.05 * max(xs.ptp(), ys.ptp())
        box = np.zeros_like(m0); box[int(max(0, ys.min() - pad)):int(ys.max() + pad) + 1, int(max(0, xs.min() - pad)):int(xs.max() + pad) + 1] = True
        near = cv2.dilate(m0.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0; m1 = m0.copy()
        for pw in parts:
            pred.handle_request(request=dict(type="reset_session", session_id=sid))
            pred.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text=pw))
            for r in pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
                if r["frame_index"] != 0: continue
                for kk in range(len(r["outputs"]["out_obj_ids"])):
                    pm = r["outputs"]["out_binary_masks"][kk].astype(bool)
                    if pm.any() and (pm & near).any() and (pm & box).sum() >= 0.5 * pm.sum(): m1 |= pm
        m1 = cv2.morphologyEx(m1.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        best = (m1, best[1]); best_cov = float(m1[sp[:, 1], sp[:, 0]].mean()) if len(sp) else 0.0
        vw["seed_cover"] = round(best_cov, 3)
    pred.handle_request(request=dict(type="close_session", session_id=sid))
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
    print(f"[mask_views] view {vw['k']} ({vw['name']}): {vw['instances']} instance(s), seed cover {best_cov:.2f}"
          + (f", score {vw['score']}" if vw.get("mask") else ", NO MASK"), flush=True)
json.dump(meta, open(f"{run}/views.json", "w"), indent=1)
