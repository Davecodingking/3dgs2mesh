# Step views: several views of the object rendered from the 3DGS, for the multi-view steps (vmask, vote, fit, compare).
# The object's seed points are the scene points inside the view's mask (front surface + a little behind it). Cameras look
# at their centre from the panorama camera's side (the back of an object was never seen, renders there are mush): azimuth /
# height pairs in VIEWS, distance from the object's width. Two more views:
#   given   the object's own view (crop.jpg + mask_final.png): it already has its mask
#   photo   when the object's view is a 3DGS render (camera not at the panorama centre): the panorama photo looking at the
#           object, the sharpest view there is; vmask masks it
# both weigh double in the vote and the fit. Per view writes views/<k>/frames/0000.jpg (SAM3 reads a folder), depth.npy
# (expected depth), cam.json, seed_px.npy (seed points projected into it: vmask keeps the SAM3 instance covering most of
# them); and views.json + seed.npy for the later steps.
import os, json, shutil, numpy as np, cv2
from PIL import Image
from .common import load_cam, read_ply, xyz, opacity, gs_tensors, render, look_at, focal, pano_crop

VIEWS = [(-45, "mid"), (0, "mid"), (45, "mid"), (-20, "high"), (20, "high"), (0, "low")]   # azimuth from the pano side
S, FOV = 1024, 60.0


def run(c):
    out = c.dir; f = focal(FOV, S)
    cam0 = load_cam(out); Rb0, C0, f0, S0 = cam0["Rb"], cam0["C"], cam0["f"], cam0["size"]
    mask0 = np.array(Image.open(f"{out}/mask_final.png").convert("L")) > 127
    v = read_ply(c.ply); P = xyz(v); op = opacity(v)
    H = max(0.05, float(np.percentile(P[op > 0.4, 1], 97)))     # camera height above the floor (place's unit)
    floor_y = H                                                  # the panorama camera is at y = 0, y is down

    Q = (P - C0) @ Rb0; z = Q[:, 2]; ok = (z > 0.05) & (op > 0.4)
    u = np.full(len(P), -1, np.int64); w = np.full(len(P), -1, np.int64)
    u[ok] = np.floor(f0 * Q[ok, 0] / z[ok] + S0 / 2); w[ok] = np.floor(f0 * Q[ok, 1] / z[ok] + S0 / 2)
    ok &= (u >= 0) & (u < S0) & (w >= 0) & (w < S0)
    inm = np.zeros(len(P), bool); inm[ok] = mask0[w[ok], u[ok]]
    hist, edges = np.histogram(z[inm], bins=np.arange(0.1 * H, 6 * H, 0.05 * H))
    Zf = edges[int(np.argmax(hist >= 0.25 * hist.max()))] + 0.025 * H
    seed = P[inm & (z > Zf - 0.08 * H) & (z < Zf + 0.8 * H)]
    center = np.median(seed, 0)
    r = 0.5 * max(np.percentile(seed[:, k], 98) - np.percentile(seed[:, k], 2) for k in (0, 2))
    dist = float(np.clip(2.6 * r + 0.15, 0.4, 1.2))
    heights = dict(low=0.12, mid=0.5 * H, high=H)
    base = C0 - center; base[1] = 0; base /= np.linalg.norm(base)

    GS = gs_tensors(v); views = []
    if os.path.isdir(f"{out}/views"): shutil.rmtree(f"{out}/views")   # a rerun must not keep masks of old views

    def add(k, img_bgr, depth, cam):
        d = f"{out}/views/{k}"; os.makedirs(f"{d}/frames", exist_ok=True)
        if img_bgr is not None: cv2.imwrite(f"{d}/frames/0000.jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.save(f"{d}/depth.npy", depth.astype(np.float32))
        json.dump(cam, open(f"{d}/cam.json", "w"), indent=1)
        Rb, C, fk, Sk = np.array(cam["Rb"]), np.array(cam["C"]), cam["f"], cam["size"]
        sq = (seed - C) @ Rb; sp = np.stack([fk * sq[:, 0] / sq[:, 2] + Sk / 2, fk * sq[:, 1] / sq[:, 2] + Sk / 2], 1)
        np.save(f"{d}/seed_px.npy", sp[sq[:, 2] > 0.05].astype(np.float32))
        views.append(dict(k=k, **cam))

    for k, (az, hn) in enumerate(VIEWS):
        t = np.radians(az); cs, sn = np.cos(t), np.sin(t)
        side = np.array([cs * base[0] + sn * base[2], 0.0, -sn * base[0] + cs * base[2]])
        C = center + dist * side; C[1] = floor_y - heights[hn]
        Rb = look_at(C, center)
        rgb, depth, _ = render(GS, Rb, C, f, S)
        add(k, rgb[..., ::-1], depth, dict(fov=FOV, size=S, f=f, Rb=Rb.tolist(), C=C.tolist(), azimuth=az, height=heights[hn],
                                           name=f"{az:+d} {hn}"))
    # given: the object's own view, camera and mask exist; only the depth is rendered
    _, depth, _ = render(GS, Rb0, C0, f0, S0)
    k = len(views); os.makedirs(f"{out}/views/{k}/frames", exist_ok=True)
    shutil.copy(f"{out}/crop.jpg", f"{out}/views/{k}/frames/0000.jpg"); shutil.copy(f"{out}/mask_final.png", f"{out}/views/{k}/mask.png")
    add(k, None, depth, dict(fov=cam0["fov"], size=S0, f=f0, Rb=Rb0.tolist(), C=C0.tolist(), name=f"given {c.run}",
                             given=True, weight=2.0, mask=True))
    if C0.any():                                                 # photo: the panorama looking at the object
        Rb = look_at(np.zeros(3), center)
        fov = float(np.clip(2.5 * np.degrees(2 * np.arctan(r / np.linalg.norm(center))), 40.0, 90.0)); fp = focal(fov, S)
        _, depth, _ = render(GS, Rb, np.zeros(3), fp, S)
        add(len(views), pano_crop(c.pano, Rb, fp, S), depth, dict(fov=fov, size=S, f=fp, Rb=Rb.tolist(), C=[0.0, 0.0, 0.0],
                                                                  name="photo", photo=True, weight=2.0))
    np.save(f"{out}/seed.npy", seed.astype(np.float32))
    json.dump(dict(source=c.run, center=center.tolist(), radius=r, dist=dist, unit=H, floor_y=floor_y, views=views),
              open(f"{out}/views.json", "w"), indent=1)
    print(f"[views] {len(seed)} seed points, centre {center.round(3)}, half width {r:.3f}, camera distance {dist:.2f}, "
          f"{len(views)} views ({', '.join(x['name'] for x in views)}) -> {out}/views/")
