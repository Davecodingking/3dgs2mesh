# Step check: look at the replacement. Renders the 3DGS (gsplat) and the placed mesh (pyrender, unlit: its texture is
# baked from the photo) and composites them by depth.
#   rows    view camera / camera moved 0.3 to the left / 0.3 to the right, both looking at the object
#   columns photo (view camera only) | 3DGS before | 3DGS after deleting | after deleting + mesh | what was deleted
# Also: mesh silhouette vs SAM3 mask in the view camera (IoU) -> check_silhouette.jpg; numbers in check.json.
import os, json, numpy as np, cv2
os.environ["PYOPENGL_PLATFORM"] = "egl"
from PIL import Image, ImageDraw
from .common import load_cam, read_ply, gs_tensors, render, look_at, font

T = 512                                                       # tile size


def run(c):
    import trimesh, pyrender
    cam = load_cam(c.dir); S, f = int(cam["size"]), float(cam["f"]); Rb, C0 = cam["Rb"], cam["C"]
    renderer = pyrender.OffscreenRenderer(S, S)
    meshes = trimesh.load(f"{c.dir}/gen/{c.name}_world.glb", force="scene").dump()

    def render_mesh(Rcw, C):
        sc = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0])
        for m in meshes: sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        pose = np.eye(4); pose[:3, :3] = Rcw @ np.diag([1.0, -1.0, -1.0]); pose[:3, 3] = C      # OpenCV -> OpenGL camera
        sc.add(pyrender.IntrinsicsCamera(fx=f, fy=f, cx=S / 2, cy=S / 2, znear=0.01, zfar=100.0), pose=pose)
        col, dep = renderer.render(sc, flags=pyrender.RenderFlags.FLAT)
        return col[..., :3], dep

    gs_before, gs_after, gs_removed = (gs_tensors(read_ply(p)) for p in (c.ply, f"{c.dir}/scene_carved.ply", f"{c.dir}/removed.ply"))
    V = np.concatenate([m.vertices for m in meshes]); center = (V.min(0) + V.max(0)) / 2
    side = Rb[:, 0] * np.array([1.0, 0.0, 1.0]); side /= np.linalg.norm(side)
    fnt = font(22); zh = fnt is not None
    views = [("切图相机" if zh else "view camera", Rb, C0)]
    for label, sgn in ((("相机左移 0.3" if zh else "camera 0.3 left"), -1), (("相机右移 0.3" if zh else "camera 0.3 right"), 1)):
        C = C0 + sgn * 0.3 * side; views.append((label, look_at(C, center), C))
    heads = ["原图", "原来的 3DGS", "删点后", "删点后 + 网格", "删掉的点"] if zh else ["photo", "3DGS before", "after", "after + mesh", "deleted"]

    rows = []; nums = {}
    for vi, (label, Rcw, C) in enumerate(views):
        rb, _, _ = render(gs_before, Rcw, C, f, S)
        ra, da, aa = render(gs_after, Rcw, C, f, S)
        mc, md = render_mesh(Rcw, C)
        hit = md > 0
        on_top = hit & ((md < da) | (aa < 0.5))
        comp = ra.copy(); comp[on_top] = mc[on_top]
        rr, _, _ = render(gs_removed, Rcw, C, f, S)           # what was deleted: anything besides the object is collateral
        first = cv2.imread(f"{c.dir}/crop.jpg")[..., ::-1] if vi == 0 else np.full((S, S, 3), 255, np.uint8)
        tiles = [Image.fromarray(np.ascontiguousarray(t)).resize((T, T)) for t in (first, rb, ra, comp, rr)]
        for i, (t, hd) in enumerate(zip(tiles, heads)):
            if i == 0 and vi: continue                        # the photo column is empty below the first row
            d = ImageDraw.Draw(t); d.rectangle([0, 0, (24 if zh else 12) * len(hd) + 16, 34], fill=(0, 0, 0))
            d.text((8, 4), hd, font=fnt, fill=(255, 255, 255))
        if vi: ImageDraw.Draw(tiles[0]).text((20, T // 2 - 14), label, font=fnt, fill=(0, 0, 0))
        else: ImageDraw.Draw(tiles[0]).text((8, T - 34), label, font=fnt, fill=(255, 255, 0))
        rows.append(np.concatenate([np.asarray(t) for t in tiles], 1))
        if vi == 0:                                          # silhouette vs mask in the view camera
            mask = np.array(Image.open(f"{c.dir}/mask_final.png").convert("L")) > 127
            iou = float((hit & mask).sum() / max(1, (hit | mask).sum()))
            nums.update(silhouette_iou=iou, mesh_px=int(hit.sum()), mask_px=int(mask.sum()))
            ov = cv2.imread(f"{c.dir}/crop.jpg")
            for m_, colr in ((mask, (0, 255, 255)), (hit, (0, 0, 255))):
                cs, _ = cv2.findContours(m_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(ov, cs, -1, colr, 3)
            cv2.imwrite(f"{c.dir}/check_silhouette.jpg", ov)
    Image.fromarray(np.concatenate(rows, 0)).save(f"{c.dir}/check.jpg", quality=90)
    json.dump(nums, open(f"{c.dir}/check.json", "w"), indent=1)
    print(f"[check] silhouette IoU {nums['silhouette_iou']:.3f} (mesh {nums['mesh_px']} px, mask {nums['mask_px']} px) "
          f"-> {c.dir}/check.jpg, {c.dir}/check_silhouette.jpg")
