# Compose a scene: every finished object of it together, for the web viewer. Applies every object's deletion (carve's
# label_drop, hand edits included) to the original 3DGS and writes into the scene's compose dir:
#   scene.ply      the 3DGS with all those objects deleted (the deliverable, with the meshes)
#   scene.splat    the same, pruned (alpha < 0.02 dropped) in the 32-byte .splat format the viewer loads
#   orig.splat     the untouched 3DGS, same format, to compare
#   <run>.glb      each object's placed mesh (gen/<name>_world.glb), in the 3DGS frame
#   objects.json   the objects (run, name, label, glb, centre) and a point to orbit around (middle of the objects;
#                  1 in front of the panorama camera when there are none)
# c: scene, ply, out, objects = [{dir, run, name, label}]  (pipeline.py picks the objects and holds the scene's lock)
import os, glob, json, shutil, numpy as np
from .common import read_ply, write_ply, to_splat
from .carve import label_drop


def run(c):
    import trimesh
    os.makedirs(c.out, exist_ok=True)
    for old in glob.glob(f"{c.out}/*.glb"): os.remove(old)
    v = read_ply(c.ply)
    drop = np.zeros(len(v), bool); objects = []
    for o in c.objects:
        d = label_drop(v, o["dir"])[0]
        print(f"[compose] {o['run']}: deletes {int(d.sum())} gaussians ({int((d & drop).sum())} already deleted by others)")
        drop |= d
        glb = f"{o['run']}.glb"; shutil.copy(f"{o['dir']}/gen/{o['name']}_world.glb", f"{c.out}/{glb}")
        b = trimesh.load(f"{c.out}/{glb}", force="mesh").bounds
        objects.append(dict(run=o["run"], name=o["name"], label=o.get("label", ""), glb=glb, center=((b[0] + b[1]) / 2).tolist()))
    write_ply(v[~drop], f"{c.out}/scene.ply")
    n1 = to_splat(v[~drop], f"{c.out}/scene.splat"); n0 = to_splat(v, f"{c.out}/orig.splat")
    target = np.mean([o["center"] for o in objects], 0).tolist() if objects else [0.0, 0.0, 1.0]
    json.dump(dict(scene=c.scene, objects=objects, target=target, frame="OpenCV world: x right, y down, z forward; camera at origin"),
              open(f"{c.out}/objects.json", "w"), indent=1)
    print(f"[compose] {c.scene}: {len(objects)} object(s), deleted {int(drop.sum())} of {len(v)}; scene.splat {n1} splats "
          f"({n1 * 32 / 2**20:.1f} MB), orig.splat {n0} -> {c.out}")
