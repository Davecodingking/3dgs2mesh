# Experiment: measure the old object and refine the placed mesh against all views (steps 5-7 of the multi-view flow).
# Starts from place.py's single-view placement, rescaled so the mesh's footprint matches the measured old object
# ("rough placement"). In the mesh's own heading frame (along / across / up) it then adjusts
#   yaw +-15 deg, uniform scale 0.8-1.3, a stretch per axis 0.9-1.1 (the +-10% agreed on 10-14), the horizontal position;
#   the height follows from landing: the lowest vertex stays on the floor place.py measured (rest = floor / table).
# Cost = (1 - IoU) between the mesh's real outline (rasterised with nvdiffrast) and each view's mask (holes filled: a tray
#   on a table hides part of it), averaged with the view weights (given views, the photo crop and the generation view,
#   count double) after dropping the single worst rendered view (renders from far off the panorama camera are blurry and
#   SAM3 follows the blur: A01 table, view +45 mid), + how far the old object's vote points are from the mesh.
# "Measure" = the old object's length / width / height from its vote points (2-98 %, heading frame). A stretch that ends
#   at the 10% limit is reported: SAM 3D's proportions are off.
# Writes gen/<name>_world.glb (the single-view one is kept as gen/<name>_world_single.glb) and fit_mv.json; updates the
# transforms in place.json.
# usage: fit_mv.py <run> <scene.ply> <name>
import os, sys, json, shutil, numpy as np, trimesh, cv2, torch
import nvdiffrast.torch as dr
from PIL import Image
from plyfile import PlyData

run, scene_ply, name = sys.argv[1:4]
G = 256                                                       # silhouette resolution per view
dev = "cuda"
P = json.load(open(f"{run}/place.json")); meta = json.load(open(f"{run}/views.json"))
floor_y, H = P["metrics"]["floor_y"], P["unit"]
single = f"{run}/gen/{name}_world_single.glb"
if not os.path.exists(single): shutil.copy(f"{run}/gen/{name}_world.glb", single)
# start from the single-view placement: the posed mesh mapped by the transform that produced <name>_world_single.glb
posed = trimesh.load(f"{run}/gen/{name}_posed.glb", force="mesh", process=False)
Vs = np.asarray(trimesh.load(single, force="mesh", process=False).vertices)
A = np.linalg.lstsq(np.c_[np.asarray(posed.vertices), np.ones(len(Vs))], Vs, rcond=None)[0]
A0 = np.eye(4); A0[:3, :] = A.T
geo = trimesh.Trimesh(vertices=Vs, faces=np.asarray(posed.faces), process=True)        # welded, for outline and landing
VM = np.asarray(geo.vertices, float); FM = np.asarray(geo.faces, np.int32)

phi = P["carve"]["phi"]                                       # heading frame of the placed mesh, up = -y
E = np.stack([[np.cos(phi), 0, np.sin(phi)], [-np.sin(phi), 0, np.cos(phi)], [0, -1.0, 0]], 1)
c0 = (VM.min(0) + VM.max(0)) / 2
def extents(X): L = (X - c0) @ E; return np.percentile(L, 98, 0) - np.percentile(L, 2, 0)

v = PlyData.read(scene_ply)["vertex"].data
labels = np.load(f"{run}/labels.npy")
op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
OBJ = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)[(labels == 1) & (op > 0.3)]
OBJs = OBJ[np.random.default_rng(7).choice(len(OBJ), min(len(OBJ), 4000), replace=False)]

old = extents(OBJ); before = extents(VM)
s_rough = float(np.clip(np.sqrt((old[0] / before[0]) * (old[1] / before[1])), 0.85, 1.2))   # footprint area match

views, Ms, Ws, given = [], [], [], []
for vw in meta["views"]:
    if not vw.get("mask"): continue
    M = np.array(Image.open(f"{run}/views/{vw['k']}/mask.png").convert("L")) > 127
    ff = np.zeros((M.shape[0] + 2, M.shape[1] + 2), np.uint8); outside = M.astype(np.uint8).copy()
    cv2.floodFill(outside, ff, (0, 0), 2); M = outside != 2                       # fill holes
    Ms.append(cv2.resize(M.astype(np.uint8), (G, G), interpolation=cv2.INTER_AREA) > 0)
    views.append((np.array(vw["Rb"]), np.array(vw["C"]), vw["f"] * G / vw["size"], vw["name"]))
    Ws.append(float(vw.get("weight", 1.0))); given.append(bool(vw.get("given")))
Rb_t = torch.tensor(np.stack([x[0] for x in views]), dtype=torch.float32, device=dev)
C_t = torch.tensor(np.stack([x[1] for x in views]), dtype=torch.float32, device=dev)
f_t = torch.tensor([x[2] for x in views], dtype=torch.float32, device=dev)
M_t = torch.tensor(np.stack(Ms), device=dev); W_np = np.array(Ws); given = np.array(given)
glctx = dr.RasterizeCudaContext()
tri = torch.tensor(FM, dtype=torch.int32, device=dev)
VM_t = torch.tensor(VM, dtype=torch.float32, device=dev); OBJ_t = torch.tensor(OBJs, dtype=torch.float32, device=dev)
E_t = torch.tensor(E, dtype=torch.float32, device=dev); c0_t = torch.tensor(c0, dtype=torch.float32, device=dev)

def transform_t(p, X):
    """p = [yaw deg, scale, stretch along, across, up, dx, dz]; then the lowest vertex is put on the floor."""
    t = np.radians(p[0]); c, s = np.cos(t), np.sin(t)
    Ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32, device=dev)
    St = E_t @ torch.diag(torch.tensor([p[1] * p[2], p[1] * p[3], p[1] * p[4]], dtype=torch.float32, device=dev)) @ E_t.T
    Y = (X - c0_t) @ (Ry @ St).T + c0_t
    return Y + torch.tensor([p[5], 0.0, p[6]], dtype=torch.float32, device=dev)
def placed(p):
    Y = transform_t(p, VM_t); Y[:, 1] += floor_y - Y[:, 1].max(); return Y
def outline_ious(Y):
    Q = torch.einsum("kij,kni->knj", Rb_t, Y[None].expand(len(views), -1, -1) - C_t[:, None, :])   # camera coords per view
    z = Q[..., 2]
    clip = torch.stack([f_t[:, None] * Q[..., 0] / (G / 2), f_t[:, None] * Q[..., 1] / (G / 2), torch.zeros_like(z), z], -1)
    rast, _ = dr.rasterize(glctx, clip.contiguous(), tri, resolution=[G, G])
    sil = rast[..., 3] > 0
    return ((sil & M_t).sum((1, 2)).float() / (sil | M_t).sum((1, 2)).clamp(min=1).float()).cpu().numpy()
def cost(p, detail=False):
    if abs(p[0]) > 15 or not 0.8 <= p[1] <= 1.3 or any(not 0.9 <= x <= 1.1 for x in p[2:5]): return 9.0
    Y = placed(p); ious = outline_ious(Y)
    keep = np.ones(len(ious), bool)
    rendered = np.nonzero(~given)[0]
    if len(rendered) >= 3: keep[rendered[np.argmin(ious[rendered])]] = False                 # drop the worst rendered view
    sil = float(np.sum(W_np[keep] * (1 - ious[keep])) / np.sum(W_np[keep]))
    d = torch.cdist(OBJ_t, Y).min(1).values; d3 = float(torch.sort(d).values[: int(0.9 * len(d))].mean()) / H
    c = sil + d3
    return (c, ious, d3) if detail else c

p = [0.0, s_rough, 1.0, 1.0, 1.0, 0.0, 0.0]
c_single, iou_single, d_single = cost([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], True)
cst = cost(p); c_start, iou_start, d_start = cost(p, True)
steps0 = [6.0, 0.06, 0.04, 0.04, 0.04, 0.03 * H, 0.03 * H]
for rs in range(7):
    steps = [st / (1 + rs) for st in steps0]; prev = cst
    for _ in range(6):
        for k in range(7):
            for sg in (-1, 1):
                q = list(p); q[k] += sg * steps[k]; cq = cost(q)
                if cq < cst: cst, p = cq, q
        steps = [st * 0.7 for st in steps]
    if prev - cst < 1e-4: break
c_end, iou_end, d_end = cost(p, True)
if c_end >= c_single:                                         # never worse than the single-view placement
    p, c_end, iou_end, d_end = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], c_single, iou_single, d_single

# the refined placement as an affine map from the posed mesh, applied to the textured file
Yb = placed(p).cpu().numpy()
B = np.linalg.lstsq(np.c_[VM, np.ones(len(VM))], Yb, rcond=None)[0]       # single-view world -> refined world (affine)
Aw = np.eye(4); Aw[:3, :] = B.T
posed_to_world = Aw @ A0
meshes = trimesh.load(f"{run}/gen/{name}_posed.glb", force="scene").dump()
for m in meshes: m.apply_transform(posed_to_world)
trimesh.Scene(meshes).export(f"{run}/gen/{name}_world.glb")
Vg = np.asarray(trimesh.load(f"{run}/gen/{name}.glb", force="mesh", process=False).vertices)
Vw = np.asarray(posed.vertices) @ posed_to_world[:3, :3].T + posed_to_world[:3, 3]
Af = np.linalg.lstsq(np.c_[Vg, np.ones(len(Vg))], Vw, rcond=None)[0]; glb_to_world = np.eye(4); glb_to_world[:3, :] = Af.T
P["posed_to_world"] = posed_to_world.tolist(); P["glb_to_world"] = glb_to_world.tolist()
P["fit_mv"] = dict(params=dict(yaw_deg=p[0], scale=p[1], stretch_along=p[2], stretch_across=p[3], stretch_up=p[4], dx=p[5], dz=p[6]))
json.dump(P, open(f"{run}/place.json", "w"), indent=1)

after = extents(Yb)
at_limit = [n for n, x in zip(("along", "across", "up"), p[2:5]) if abs(x - 1) > 0.095]
names = [x[3] for x in views]
out = dict(old_object_size=old.tolist(), mesh_size_single=before.tolist(), mesh_size_after=after.tolist(), rough_scale=s_rough,
           params=P["fit_mv"]["params"], views=names, weights=W_np.tolist(), iou_single=iou_single.tolist(), iou_after=iou_end.tolist(),
           point_dist_single=d_single, point_dist_after=d_end, stretch_at_limit=at_limit)
json.dump(out, open(f"{run}/fit_mv.json", "w"), indent=1)
f3 = lambda a: " x ".join(f"{x:.3f}" for x in a)
print(f"[fit_mv] measured old object (along x across x up): {f3(old)}; rough scale {s_rough:.3f}")
print(f"[fit_mv] mesh single-view {f3(before)}  ->  after {f3(after)}")
print(f"[fit_mv] yaw {p[0]:+.1f} deg, scale {p[1]:.3f}, stretch {p[2]:.3f} / {p[3]:.3f} / {p[4]:.3f}, move {p[5]:+.3f} / {p[6]:+.3f}"
      + (f"  ** stretch at the 10% limit on {at_limit}: SAM 3D's proportions are off" if at_limit else ""))
print(f"[fit_mv] outline IoU per view {names}:\n         single {np.round(iou_single, 2).tolist()}\n         after  {np.round(iou_end, 2).tolist()}; "
      f"old points to mesh {d_single:.4f} -> {d_end:.4f} H; cost {c_single:.4f} -> {c_end:.4f}")
