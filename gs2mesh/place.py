# Step place: put the SAM 3D mesh into the 3DGS scene where the original object stands, from the view camera alone.
# Python port of solve() in the colleague's place.mjs: same costs, constants and search. Changes agreed on 10-14:
#   1. the object's scene points are the ones that project inside the SAM3 mask, not inside a box
#      (the mask's bounding box still plays the box's other two roles: silhouette term, "just below" floor probe)
#   2. landing follows where the object rests: floor = his floor probe + ring re-measure; table = probe only
#      (the ring around a table-top object reaches down to the floor); hung is not supported yet
#   3. the asset points are the mesh itself, not SAM 3D's gaussians. Thin parts (legs, arm edges) are a tiny share of the
#      gaussians, and the 1-99 % outline / 99.8 % lowest point in place.mjs cut them off: the fit matched the gaussians'
#      inner 98 % to the mask box, so the mesh came out about 6 % too wide per side, with its legs 8 % of H through the floor.
#      Surface samples (even by area) drive the plane, the width and the chamfer terms; all vertices give the true lowest
#      point (floor contact) and the true outline (silhouette term).
# Search: start from SAM 3D's own heading, level the mesh's largest horizontal plane, then adjust yaw +-30 deg, scale
# +-35 %, x and z; the lowest vertex always sits on the floor. Cost = scene -> mesh distance + mesh (seen side) -> scene
# distance + 0.5 x outline box vs mask box + a small yaw penalty.
# The asset is SAM 3D's mesh posed in its own camera frame (gen: <name>_posed.glb).
# Writes place.json (transforms, the heading frame later steps use, metrics) and gen/<name>_world.glb (mesh in the 3DGS frame;
# also as <name>_world_single.glb, which fit starts from).
import sys, json, shutil, numpy as np
from .common import load_cam, read_ply, xyz, sigmoid

W_SIL, W_YAW, YAW_MAX, SCALE_BAND = 0.5, 0.03, 30.0, (0.65, 1.35)     # as in place.mjs


def pct(a, p):
    """Same percentile as place.mjs: sort, take the rounded rank."""
    a = np.sort(np.asarray(a, float).ravel()); n = len(a)
    return a[min(n - 1, max(0, int(np.floor(p / 100 * (n - 1) + 0.5))))]
def trimmed_mean(a, frac):
    s = np.sort(a); return s[:max(1, int(np.floor(frac * len(s))))].mean()
def pick(rng, n, k): return np.arange(n) if n <= k else rng.choice(n, k, replace=False)
def ryaw(deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rodrigues(ax, a):
    l = np.linalg.norm(ax)
    if l < 1e-9: return np.eye(3)
    x, y, z = ax / l; K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
def project(P, cam):
    """World points -> pixel u, v and depth z in the view camera."""
    C = (P - cam["C"]) @ cam["Rb"]; z = C[:, 2]; f, size = cam["f"], cam["size"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return f * C[:, 0] / z + size / 2, f * C[:, 1] / z + size / 2, z


def run(c):
    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree
    run_dir, name, rest = c.dir, c.name, c.rest
    if rest not in ("floor", "table"):
        sys.exit(f"[place] rest={rest!r}: only floor / table are supported so far")
    rng = np.random.default_rng(7)

    # ------------------------------------------------------------ camera, mask
    cam = load_cam(run_dir); Rb = cam["Rb"]; size = int(cam["size"])
    Bx = Rb[:, 0]
    uc = Rb.T @ np.array([0.0, -1.0, 0.0])                    # world up in view-camera axes
    mask = np.array(Image.open(f"{run_dir}/mask_final.png").convert("L")) > 127
    mv_, mu_ = np.nonzero(mask)
    x0, y0, x1, y1 = float(mu_.min()), float(mv_.min()), float(mu_.max() + 1), float(mv_.max() + 1)

    # ------------------------------------------------------------ scene evidence
    sv = read_ply(c.ply)
    SP = xyz(sv)[sigmoid(sv["opacity"]) > 0.4]                 # float32 sigmoid, as the original
    H = max(0.05, pct(SP[:, 1], 97))                          # length unit: about the camera height
    u, v, z = project(SP, cam)
    front = z > 0.05
    ui = np.where(front, np.floor(np.nan_to_num(u, nan=-1, posinf=-1, neginf=-1)), -1).astype(np.int64)
    vi = np.where(front, np.floor(np.nan_to_num(v, nan=-1, posinf=-1, neginf=-1)), -1).astype(np.int64)
    inimg = front & (ui >= 0) & (ui < size) & (vi >= 0) & (vi < size)
    inmask = np.zeros(len(SP), bool); inmask[inimg] = mask[vi[inimg], ui[inimg]]
    idx0 = np.nonzero(inmask)[0]                              # change 1: inside the mask, not inside a box
    if len(idx0) < 300:
        sys.exit(f"[place] only {len(idx0)} scene points fall inside the mask: wrong camera or not on an object")
    lo, bw = 0.1 * H, 0.05 * H; nb = int(np.ceil((6 * H - lo) / bw))
    hb = np.floor((z[idx0] - lo) / bw).astype(np.int64)
    hist = np.bincount(hb[(hb >= 0) & (hb < nb)], minlength=nb)
    Zf = lo + (int(np.argmax(hist >= 0.25 * hist.max())) + 0.5) * bw          # front surface of what is in the mask
    full = idx0[z[idx0] < Zf + 1.5 * H]
    shell = full[(z[full] > Zf - 0.08 * H) & (z[full] < Zf + 0.8 * H)]
    below = SP[front & (u >= x0) & (u <= x1) & (v > y1 + 5) & (v < y1 + 90), 1]
    floorY = pct(below, 50) if len(below) > 20 else H
    S = SP[shell[pick(rng, len(shell), 1500)]]; SF = SP[full[pick(rng, len(full), 2500)]]
    if len(S) < 50: sys.exit("[place] too few scene points on the object")
    print(f"[place] unit {H:.3f}, front depth {Zf:.3f}, floor y {floorY:.3f}, {len(shell)} shell / {len(full)} frustum points", flush=True)

    # ------------------------------------------------------------ asset: the mesh, posed in SAM 3D's camera frame (x left, y up)
    mesh = trimesh.load(f"{run_dir}/gen/{name}_posed.glb", force="mesh")
    # geometry only, vertices merged by position: the textured mesh repeats vertices along UV seams, so splitting it as
    # loaded cuts it into its UV islands (48k faces -> dozens of pieces), and "drop small pieces" then throws away real parts
    geo = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=True)
    parts = [m for m in geo.split(only_watertight=False) if len(m.faces) >= 0.01 * len(geo.faces)]     # drop floating bits
    body = trimesh.util.concatenate(parts) if len(parts) > 1 else (parts[0] if parts else geo)
    SM = np.asarray(trimesh.sample.sample_surface(body, 20000, seed=7)[0], float)    # change 3: even surface samples
    VM = np.asarray(body.vertices, float)                                           # and every vertex
    ext = max(pct(SM[:, k], 99) - pct(SM[:, k], 1) for k in range(3))
    ctr = SM.mean(0)
    FL = np.diag([-1.0, -1.0, 1.0])                           # rotate 180 deg about z: SAM 3D camera -> OpenCV camera
    Qs = (SM - ctr) @ FL.T                                    # F (q - c)
    Qv = (VM - ctr) @ FL.T
    print(f"[place] mesh: {len(mesh.faces)} faces, kept {len(parts)} part(s) / {len(body.faces)} faces for the fit", flush=True)

    # ------------------------------------------------------------ gravity: the best horizontal plane
    best_cnt, best_n = 0, None; Qh = Qs[::2]
    for _ in range(600):
        p3 = Qs[rng.integers(0, len(Qs), 3)]
        nv = np.cross(p3[1] - p3[0], p3[2] - p3[0]); l = np.linalg.norm(nv)
        if l < 1e-9: continue
        nv = nv / l
        if abs(nv @ uc) < 0.85: continue
        cnt = int((np.abs((Qh - p3[0]) @ nv) < 0.01 * ext).sum())
        if cnt > best_cnt: best_cnt, best_n = cnt, (nv if nv @ uc > 0 else -nv)
    tilt = float(np.degrees(np.arccos(np.clip(best_n @ uc, -1, 1)))) if best_n is not None else 0.0
    print(f"[place] horizontal plane: {best_cnt * 2} inliers, tilt {tilt:.1f} deg", flush=True)
    Sx = S @ Bx; sw = pct(Sx, 98) - pct(Sx, 2)
    medS = [pct(S[:, k], 50) for k in range(3)]
    treeSF = cKDTree(SF)

    def fit(use_up):
        Rup = rodrigues(np.cross(best_n, uc), np.radians(tilt)) if (use_up and best_n is not None) else np.eye(3)
        O0 = Qs @ (Rb @ Rup).T                                # world axes, gravity aligned, centred
        OV = Qv @ (Rb @ Rup).T
        OVy = float(OV[:, 1].max())                           # lowest vertex (y down); yaw about the vertical keeps it
        sub = O0[pick(rng, len(O0), 3000)]; floor = [floorY]
        Ox = O0 @ Bx; s0 = min(3.0, max(0.1, sw / (pct(Ox, 98) - pct(Ox, 2))))

        def place(p, pts):                                    # yaw about the vertical, scale, x/z, lowest vertex on the floor
            W = p[1] * (pts @ ryaw(p[0]).T); W[:, 0] += p[2]; W[:, 2] += p[3]
            W[:, 1] += floor[0] - p[1] * OVy; return W
        last = [0.0, 0.0, 0.0]

        def cost(p):
            if abs(p[0]) > YAW_MAX or p[1] < SCALE_BAND[0] * s0 or p[1] > SCALE_BAND[1] * s0: return 1e3
            W = place(p, sub)
            c1 = trimmed_mean(cKDTree(W).query(S)[0], 0.7) / H              # scene surface -> asset
            pu, pv, pz = project(W, cam); ok = pz > 0.05
            cu = np.clip(np.floor(np.nan_to_num(pu / 32, nan=0, posinf=31, neginf=0)), 0, 31).astype(np.int64)
            cv = np.clip(np.floor(np.nan_to_num(pv / 32, nan=0, posinf=31, neginf=0)), 0, 31).astype(np.int64)
            cc = cu * 32 + cv; cell = np.full(1024, np.inf); np.minimum.at(cell, cc[ok], pz[ok])
            vis = ok & (pz <= cell[cc] + 0.02 * H)                            # asset side the camera sees
            c2 = trimmed_mean(treeSF.query(W[vis])[0], 0.7) / H if vis.sum() > 50 else 1.0
            qu, qv, qz = project(place(p, OV), cam); okv = qz > 0.05         # change 3: the whole outline, legs included
            us, vs = qu[okv], qv[okv]
            c3 = (abs(us.min() - x0) + abs(vs.min() - y0) + abs(us.max() - x1) + abs(vs.max() - y1)) / 4 / size
            last[:] = [c1, c2, c3]
            return c1 + c2 + W_SIL * c3 + W_YAW * (p[0] / 30) ** 2

        bestP, bestC = None, np.inf
        for th in range(-30, 31, 10):
            p = [float(th), s0, medS[0], medS[2]]; cp = cost(p)
            if cp < bestC: bestC, bestP = cp, p
        p, cst = list(bestP), bestC
        for _round in range(2):                               # round 2: floor measured around the fitted footprint
            steps0 = [10.0, 0.1 * s0, 0.1 * H, 0.1 * H]; prev = cst
            for rs in range(7):
                steps = [st / (1 + rs) for st in steps0]
                for _ in range(8):
                    for k in range(4):
                        for sg in (-1, 1):
                            q = list(p); q[k] += sg * steps[k]; cq = cost(q)
                            if cq < cst: cst, p = cq, q
                    steps = [st * 0.75 for st in steps]
                if prev - cst < 2e-4: break
                prev = cst
            Wf = place(p, sub); cx = Wf[:, [0, 2]].mean(0)
            rmax = max(float(np.max(np.hypot(Wf[:, 0] - cx[0], Wf[:, 2] - cx[1]))) * 0.8, 1e-3)
            ring = SP[(np.hypot(SP[:, 0] - cx[0], SP[:, 2] - cx[1]) < 1.3 * rmax) & (SP[:, 1] > floorY - 0.5 * H), 1]
            if rest == "floor" and len(ring) > 200:           # change 2: a table-top object keeps the probe height
                floor[0] = pct(ring, 85); cst = cost(p)
        cost(p)
        return dict(p=p, cost=cst, comp=list(last), Rup=Rup, OVy=OVy, floor=floor[0], s0=s0)

    cands = {"up": fit(True), "noup": fit(False)}
    opt = "up" if cands["up"]["cost"] <= cands["noup"]["cost"] else "noup"; R = cands[opt]
    th, s, tx, tz = R["p"]; Ry = ryaw(th)
    print(f"[place] tilt option costs up {cands['up']['cost']:.4f} / noup {cands['noup']['cost']:.4f} -> {opt}; "
          f"yaw {th:.1f} scale {s:.3f}", flush=True)
    Rg = Ry @ Rb @ R["Rup"] @ FL                              # posed-asset axes -> world:  world = s * Rg (q - c) + T
    T = np.array([tx, R["floor"] - s * R["OVy"], tz])

    # ------------------------------------------------------------ the mesh's heading frame and footprint (fit and the old
    # box carve use it; "carve" keeps its place.mjs name)
    phi = float(np.arctan2(Rg[2, 0], Rg[0, 0])); ca, sa = np.cos(phi), np.sin(phi)
    Wg = s * ((SM - ctr) @ Rg.T) + T
    L = np.stack([Wg[:, 0] * ca + Wg[:, 2] * sa, -Wg[:, 0] * sa + Wg[:, 2] * ca, -Wg[:, 1]], 1)
    lo3 = [pct(L[:, k], 0.5) for k in range(3)]; hi3 = [pct(L[:, k], 99.5) for k in range(3)]
    carve = dict(phi=phi, lo=lo3, hi=hi3, pad=[0.10, 0.10, 0.04], floor_h=-R["floor"] + 0.03 * H, top_margin=0.005 * H)
    max_size = max(pct(Wg[:, k], 99) - pct(Wg[:, k], 1) for k in range(3))

    # ------------------------------------------------------------ mesh into the 3DGS frame
    posed_to_world = np.eye(4); posed_to_world[:3, :3] = s * Rg; posed_to_world[:3, 3] = T - s * Rg @ ctr
    meshes = trimesh.load(f"{run_dir}/gen/{name}_posed.glb", force="scene").dump()
    for m in meshes: m.apply_transform(posed_to_world)
    trimesh.Scene(meshes).export(f"{run_dir}/gen/{name}_world.glb")
    shutil.copy(f"{run_dir}/gen/{name}_world.glb", f"{run_dir}/gen/{name}_world_single.glb")   # fit starts from this one
    # the same placement for the untouched SAM 3D file (<name>.glb): fit it from matching vertices and check it is exact
    Vg = np.asarray(trimesh.load(f"{run_dir}/gen/{name}.glb", force="mesh", process=False).vertices)
    Vw = np.asarray(trimesh.load(f"{run_dir}/gen/{name}_posed.glb", force="mesh", process=False).vertices)
    Vw = Vw @ posed_to_world[:3, :3].T + posed_to_world[:3, 3]
    A = np.linalg.lstsq(np.c_[Vg, np.ones(len(Vg))], Vw, rcond=None)[0]
    glb_to_world = np.eye(4); glb_to_world[:3, :] = A.T
    glb_err = float(np.abs(np.c_[Vg, np.ones(len(Vg))] @ A - Vw).max())

    bottom = float((s * ((VM - ctr) @ Rg.T) + T)[:, 1].max())
    out = dict(name=name, rest=rest, yaw_deg=th, scale=s, Rg=Rg.tolist(), c=ctr.tolist(), T=T.tolist(),
               posed_to_world=posed_to_world.tolist(), glb_to_world=glb_to_world.tolist(), glb_to_world_max_err=glb_err,
               carve=carve, max_size=max_size, unit=H,
               metrics=dict(tilt_deg=tilt, tilt_option=opt, cost=R["cost"], chamfer_scene_to_asset=R["comp"][0],
                            chamfer_asset_to_scene=R["comp"][1], silhouette_px=R["comp"][2] * size, floor_y=R["floor"],
                            bottom_minus_floor=bottom - R["floor"], size_world=[hi3[k] - lo3[k] for k in range(3)],
                            front_depth=Zf, scene_points=int(len(shell))))
    json.dump(out, open(f"{run_dir}/place.json", "w"), indent=1)
    print(f"[place] chamfer scene->asset {R['comp'][0]:.4f} H, asset->scene {R['comp'][1]:.4f} H, "
          f"silhouette {R['comp'][2] * size:.1f} px, floor y {R['floor']:.3f}; glb->world fit max err {glb_err:.2e}", flush=True)
    print(f"[place] -> {run_dir}/place.json, {run_dir}/gen/{name}_world.glb")
