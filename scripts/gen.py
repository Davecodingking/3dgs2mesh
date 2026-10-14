# Experiment: SAM 3D Objects on crop + mask, the same call as run.py --hq (simplify 0.9, seeded baking).
# Besides the textured GLB it keeps what the pipeline throws away: SAM 3D's own pose (local -> its camera frame),
# the gaussians posed in that camera frame (what 3dgs2mesh/place.mjs reads as its asset) and the mesh posed the same way.
# usage: gen.py <out_dir> <crop.jpg> <mask.png> <name> [seed=42] [gen_dir=gen]
import os, sys, json, numpy as np, torch
from PIL import Image

out, crop, mask_png, name = sys.argv[1:5]
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 42
GEN = sys.argv[6] if len(sys.argv) > 6 else "gen"
S3O = "/home/ubuntu/sam3d/sam-3d-objects"
os.chdir(S3O); sys.path.insert(0, S3O); sys.path.insert(0, f"{S3O}/notebook")
from inference import Inference, make_scene
from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils as _pu

_orig_to_glb = _pu.to_glb
def _seeded_to_glb(*a, **k):
    np.random.seed(SEED); torch.manual_seed(SEED)
    return _orig_to_glb(*a, **k)
_pu.to_glb = _seeded_to_glb

inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
image = np.array(Image.open(crop).convert("RGB")).astype(np.uint8)
mask = np.array(Image.open(mask_png).convert("L")) > 127
inference._pipeline.simplify = 0.9
o = inference._pipeline.run(inference.merge_mask_to_rgba(image, mask), None, SEED,
                            stage1_only=False, with_mesh_postprocess=True, with_texture_baking=True,
                            with_layout_postprocess=False, use_vertex_color=False,
                            stage1_inference_steps=None, pointmap=None)

d = f"{out}/{GEN}"; os.makedirs(d, exist_ok=True)
o["glb"].export(f"{d}/{name}.glb")
o["gs"].save_ply(f"{d}/{name}_local.ply")
pose = {k: o[k].detach().float().cpu().numpy().tolist() for k in ("rotation", "translation", "scale")}
json.dump(pose, open(f"{d}/{name}_pose.json", "w"), indent=1)

posed = make_scene(o)                                    # deep copy; gaussians moved into SAM 3D's camera frame
posed.save_ply(f"{d}/{name}_posed.ply")
# mesh: undo to_glb's z-up -> y-up turn (v_glb = v_local @ M), then the same local -> camera transform as the gaussians
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
V = torch.tensor(np.asarray(o["glb"].vertices) @ M.T, dtype=torch.float32, device=o["rotation"].device)
PC = SceneVisualizer.object_pointcloud(points_local=V.unsqueeze(0), quat_l2c=o["rotation"],
                                       trans_l2c=o["translation"], scale_l2c=o["scale"])
mesh_posed = o["glb"].copy(); mesh_posed.vertices = PC.points_list()[0].detach().cpu().numpy()
mesh_posed.export(f"{d}/{name}_posed.glb")

G = posed.get_xyz.detach().cpu().numpy(); Vm = np.asarray(mesh_posed.vertices)
print(f"[gen] pose {pose}")
print(f"[gen] posed gaussians bbox {G.min(0).round(3)} .. {G.max(0).round(3)}  ({len(G)})")
print(f"[gen] posed mesh      bbox {Vm.min(0).round(3)} .. {Vm.max(0).round(3)}  ({len(Vm)} verts)")
print(f"[gen] -> {d}")
