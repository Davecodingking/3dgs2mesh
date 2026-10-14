# Experiment: delete the old object (step 4 of the multi-view flow) as a closed volume, not point by point.
# The closure: the outline of the voted object points seen from above (convex hull of their largest blob, specks dropped),
# extruded from just above the floor layer (3% H) up to the object's top. Everything inside goes, whatever the vote said:
# per-point voting kept what the views could not see or saw wrongly (A01 table: the part behind the vase). The floor layer
# stays, and so does everything above the object's top (the vase, teapot and cups standing on the table).
# Floaters beside it go too (floaters()): outside the outline but within FLOAT_D of it, above the floor layer, no higher
# than its top, and not connected to anything that stays. A01 table: the floor behind its far edge had been put at
# table-top height by the 3DGS; from anywhere but the panorama camera it hung there as a brown slab.
# Hand edits from the 3D check page (<run>/manual.json: scene indices to "delete" / "keep", the point brush and the mesh
# eraser both write there) override all of the above.
# Writes the carved scene and <run>/removed.ply.
# usage: carve_labels.py <run> <scene.ply> <out.ply>        (compose.py, debug_export.py, mesh_erase.py import from here)
import os, sys, json, numpy as np, cv2
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

FLOAT_D, LINK, CELL = 0.05, 0.01, 0.005


def setup(v, run):
    labels = np.load(f"{run}/labels.npy")
    P = json.load(open(f"{run}/place.json")); floor = P["metrics"]["floor_y"]; H = P["unit"]
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    return labels, P, floor, H, X, floor - X[:, 1]                 # last: height above the floor


def closure(v, run):
    """(bool per gaussian inside the closure, outline polygon in world x/z, bottom height, top height, grid info)."""
    labels, P, floor, H, X, h = setup(v, run); obj = labels == 1
    xz = X[:, [0, 2]]; mn = xz[obj].min(0) - 2 * FLOAT_D
    n = np.ceil((xz[obj].max(0) + 2 * FLOAT_D - mn) / CELL).astype(int) + 1
    ij = np.floor((xz - mn) / CELL).astype(int); ing = (ij[:, 0] >= 0) & (ij[:, 0] < n[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < n[1])
    occ = np.zeros((n[1], n[0]), np.uint8); oi = ij[obj & ing]; occ[oi[:, 1], oi[:, 0]] = 1
    occ = cv2.morphologyEx(cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k, lab, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    blob = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) if k > 1 else occ
    pts = np.argwhere(blob > 0)[:, ::-1].astype(np.int32)        # (i, j) cells
    hull = cv2.convexHull(pts)[:, 0, :]
    grid = np.zeros_like(blob); cv2.fillConvexPoly(grid, hull, 1)
    bottom, top = 0.03 * H, float(np.percentile(h[obj], 99)) + 0.005
    inside = np.zeros(len(X), bool); inside[ing] = grid[ij[ing, 1], ij[ing, 0]] > 0
    inside &= (h > bottom) & (h < top)
    poly = (hull.astype(np.float64) + 0.5) * CELL + mn            # outline in world x / z
    return inside, poly, bottom, top, (grid, mn, n, ij, ing)


def floaters(v, run, drop, gridinfo, top):
    """Points hanging beside the deleted object (see the header). Returns a bool per gaussian."""
    labels, P, floor, H, X, h = setup(v, run)
    grid, mn, n, ij, ing = gridinfo
    r = int(np.ceil(FLOAT_D / CELL)); disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    band = (cv2.dilate(grid, disk) > 0) & (grid == 0)
    inband = np.zeros(len(X), bool); inband[ing] = band[ij[ing, 1], ij[ing, 0]]
    cand = inband & ~drop & (h > 0.03 * H) & (h < top + 0.005)
    out = np.zeros(len(X), bool)
    if not cand.any(): return out
    # supported = connected (chains of points closer than LINK) to something that stays: the floor layer or a kept point
    # outside the band / above the top. Judging support by "a kept point right below" alone deleted whole neighbours that
    # sit inside the band (A01: a plant on the window sill, a cup, a vase, the sofa's arm next to the table).
    reach = cv2.dilate(band.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    inreach = np.zeros(len(X), bool); inreach[ing] = reach[ij[ing, 1], ij[ing, 0]]
    anchor = inreach & ~drop & ~cand
    nodes = np.nonzero(cand | anchor)[0]
    pairs = cKDTree(X[nodes]).query_pairs(LINK, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(nodes), len(nodes)))
    ncomp, comp = connected_components(g, directed=False)
    anchored = np.zeros(ncomp, bool); anchored[comp[anchor[nodes]]] = True
    out[nodes[cand[nodes] & ~anchored[comp]]] = True
    return out


def label_drop(v, run):
    """Which gaussians of vertex array v to delete for <run>: the closure, the floaters beside it, then the hand edits."""
    inside, poly, bottom, top, gridinfo = closure(v, run)
    fl = floaters(v, run, inside, gridinfo, top)
    drop = inside | fl
    n_hand = (0, 0)
    if os.path.exists(f"{run}/manual.json"):
        man = json.load(open(f"{run}/manual.json"))
        de, ke = np.array(man.get("delete", []), np.int64), np.array(man.get("keep", []), np.int64)
        drop[de] = True; drop[ke] = False; n_hand = (len(de), len(ke))
    label_drop.counts = dict(closure=int(inside.sum()), floaters=int(fl.sum()), hand_delete=n_hand[0], hand_keep=n_hand[1])
    return drop, int(inside.sum()), int(fl.sum())


if __name__ == "__main__":
    run, scene_ply, out_ply = sys.argv[1:4]
    v = PlyData.read(scene_ply)["vertex"].data
    drop, _, _ = label_drop(v, run)
    PlyData([PlyElement.describe(v[~drop], "vertex")]).write(out_ply)
    PlyData([PlyElement.describe(v[drop], "vertex")]).write(f"{run}/removed.ply")
    c = label_drop.counts
    print(f"[carve_labels] removed {int(drop.sum())} of {len(v)}: inside the closure {c['closure']}, floaters beside it {c['floaters']}, "
          f"hand edits delete {c['hand_delete']} / keep {c['hand_keep']} -> {out_ply}")
