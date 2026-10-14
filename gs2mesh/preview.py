# Step preview: look at the generated textured mesh. Tiles: the view image | the mesh seen from the view camera | the mesh
# turned 90 / 180 / 270 deg about the room's vertical axis. The mesh is the one posed in SAM 3D's camera frame.
# Writes gen/<name>_preview.jpg.
import os, numpy as np
os.environ["PYOPENGL_PLATFORM"] = "egl"
from .common import load_cam


def run(c):
    import trimesh, pyrender, cv2
    cam = load_cam(c.dir); Rb = cam["Rb"]; dst = f"{c.dir}/gen/{c.name}_preview.jpg"
    S = 640; f = S / 2 / np.tan(np.radians(cam["fov"]) / 2)
    FL = np.diag([-1.0, -1.0, 1.0])                         # SAM 3D camera (x left, y up) <-> OpenCV camera (x right, y down)
    up = FL @ (Rb.T @ np.array([0.0, -1.0, 0.0]))           # room 'up' (OpenCV world -y) in SAM 3D's camera frame
    GL = np.diag([-1.0, 1.0, -1.0])                         # SAM 3D camera frame -> OpenGL camera frame (x right, y up, z back)
    meshes = trimesh.load(f"{c.dir}/gen/{c.name}_posed.glb", force="scene").dump()
    V = np.concatenate([m.vertices for m in meshes]); ctr = (V.min(0) + V.max(0)) / 2
    r = pyrender.OffscreenRenderer(S, S)
    tiles = [cv2.resize(cv2.imread(f"{c.dir}/crop.jpg"), (S, S))[..., ::-1]]
    for ang in (0, 90, 180, 270):
        R = trimesh.transformations.rotation_matrix(np.radians(ang), up, ctr)
        s = pyrender.Scene(bg_color=[0.92, 0.92, 0.92, 1.0], ambient_light=[0.45] * 3)
        for m in meshes:
            mm = m.copy(); mm.apply_transform(R); mm.vertices = mm.vertices @ GL.T
            s.add(pyrender.Mesh.from_trimesh(mm, smooth=False))
        s.add(pyrender.IntrinsicsCamera(fx=f, fy=f, cx=S / 2, cy=S / 2), pose=np.eye(4))
        s.add(pyrender.DirectionalLight(intensity=3.0), pose=np.eye(4))
        col, _ = r.render(s); tiles.append(col)
    cv2.imwrite(dst, np.concatenate(tiles, 1)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[preview] {len(meshes)} mesh(es), {len(V)} verts -> {dst}")
