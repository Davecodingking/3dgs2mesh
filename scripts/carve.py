# Experiment: delete the original object's gaussians from the 3DGS with the carve box from place.json, the same rule as
# merge() in 3dgs2mesh/place.mjs (heading-frame box + padding, floor and whatever sits above the object kept),
# but nothing is added back: the object is delivered as a mesh. Also writes the removed gaussians, to look at.
# usage: carve.py <run_dir> <scene.ply> <out.ply>        (compose.py imports carve_drop for several objects at once)
import sys, json, numpy as np
from plyfile import PlyData, PlyElement


def carve_drop(v, cv):
    """Which gaussians of vertex array v fall in the carve region cv (place.json "carve")."""
    X, Y, Z = (v[k].astype(np.float64) for k in ("x", "y", "z"))
    ca, sa = np.cos(cv["phi"]), np.sin(cv["phi"])
    lo, hi, pad = np.array(cv["lo"]), np.array(cv["hi"]), np.array(cv["pad"]); ext = hi - lo
    a, b, h = X * ca + Z * sa, -X * sa + Z * ca, -Y             # along heading, across, height
    return ((a > lo[0] - pad[0] * ext[0]) & (a < hi[0] + pad[0] * ext[0]) &
            (b > lo[1] - pad[1] * ext[1]) & (b < hi[1] + pad[1] * ext[1]) &
            (h > lo[2] - pad[2] * ext[2]) & (h < hi[2] - cv["top_margin"]) & (h > cv["floor_h"]))


if __name__ == "__main__":
    run, scene_ply, out_ply = sys.argv[1:4]
    v = PlyData.read(scene_ply)["vertex"].data
    drop = carve_drop(v, json.load(open(f"{run}/place.json"))["carve"])
    PlyData([PlyElement.describe(v[~drop], "vertex")]).write(out_ply)
    PlyData([PlyElement.describe(v[drop], "vertex")]).write(f"{run}/removed.ply")
    print(f"[carve] removed {int(drop.sum())} of {len(v)} gaussians -> {out_ply} (+ {run}/removed.ply)")
