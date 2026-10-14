#!/home/ubuntu/miniforge3/envs/sam3d/bin/python
"""3dgs2mesh: replace objects of a 3DGS scene with SAM 3D Objects meshes.

  ./pipeline.py steps                               list the steps
  ./pipeline.py new <scene> '<json>'                add an object to scenes/<scene>.json; prints {"run": ...}
  ./pipeline.py set <run> '<json>'                  change fields of an object (cam, click, prompt, parts, rest, cover, label)
  ./pipeline.py run <run> [--steps a,b | --from s] [--seed N]
                                                    run steps for one object (all by default); a run that includes
                                                    place / carve / fit recomposes its scene when it is done
  ./pipeline.py apply <run>                         after hand edits (manual.json): carve, compare, debug, recompose
  ./pipeline.py compose <scene>                     every finished object of the scene -> runs/<scene>_compose/
  ./pipeline.py remove <run>                        take an object out of its scene (its run dir stays), recompose

Scenes are listed in scenes/index.json, their objects in scenes/<scene>.json; an object's results go to runs/<run>/,
its progress to runs/<run>/status.json. Every step runs in its own process with the conda env it needs; this file only
uses the standard library, so any python 3 can run it.
"""
import argparse, collections, datetime, fcntl, json, math, os, re, signal, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV = {"sam3d": "/home/ubuntu/miniforge3/envs/sam3d/bin/python",   # SAM 3D Objects, gsplat, nvdiffrast, pyrender
       "sam3": "/home/ubuntu/miniforge3/envs/sam3/bin/python"}     # SAM3
# step: (env, GPU memory to claim in MB (0 = no GPU), what it does). The order is the run order.
STEPS = collections.OrderedDict([
    ("view",    ("sam3d", 6000,  "image to generate from: panorama crop (camera at the centre) or 3DGS render -> crop.jpg, cam.json")),
    ("align",   ("sam3d", 6000,  "3DGS rendered from the same camera next to the crop (panorama views) -> check_align.jpg")),
    ("mask",    ("sam3",  8000,  "SAM3: the object (prompt, click) + what lies on it (parts) -> mask_final.png")),
    ("gen",     ("sam3d", 22000, "SAM 3D Objects: textured mesh + its pose -> gen/<name>.glb, gen/<name>_posed.glb")),
    ("preview", ("sam3d", 2000,  "the mesh from 4 sides next to the crop -> gen/<name>_preview.jpg")),
    ("place",   ("sam3d", 0,     "single-view placement on the object's scene points -> place.json, gen/<name>_world.glb")),
    ("views",   ("sam3d", 6000,  "6 views around the object from the 3DGS + its own view (+ the photo) -> views/")),
    ("vmask",   ("sam3",  8000,  "SAM3 on every view -> views/<k>/mask.png")),
    ("vote",    ("sam3d", 0,     "which gaussians are the old object, by vote of the views -> labels.npy")),
    ("carve",   ("sam3d", 0,     "delete the old object (closed volume, floaters, hand edits) -> scene_carved.ply")),
    ("fit",     ("sam3d", 6000,  "refine the mesh against every view's mask -> gen/<name>_world.glb, fit_mv.json")),
    ("overlay", ("sam3d", 2000,  "mask vs mesh outline in every view -> views_overlay.jpg")),
    ("check",   ("sam3d", 6000,  "before / after / after + mesh from 3 cameras -> check.jpg, check.json")),
    ("compare", ("sam3d", 6000,  "per view: original / after / deleted -> views/<k>/cmp_*.jpg, compare.json")),
    ("debug",   ("sam3d", 0,     "data for the 3D check page viewer/debug.html -> debug/")),
])
RECOMPOSE = {"place", "carve", "fit"}                          # steps that change what the composed scene shows
FIELDS = ("label", "prompt", "parts", "rest", "cover", "cam", "click")
CHILD_ENV = dict(CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                 HF_HUB_OFFLINE="1", CUDA_HOME="/usr/local/cuda-12.8", TORCH_CUDA_ARCH_LIST="8.6",
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1", PYOPENGL_PLATFORM="egl")


def log(msg): print(f"[pipeline] {msg}", flush=True)
def now(): return datetime.datetime.now().astimezone().isoformat(timespec="seconds")   # with the offset: the page compares it with its own clock


# ---------------------------------------------------------------- files: scenes, objects, status (written atomically: the page polls them)
def read_json(path, default=None):
    try: return json.load(open(path))
    except FileNotFoundError: return default


def write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh: json.dump(obj, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def lock(name, wait=True):
    """An flock held until the returned file is closed (or the process ends). None when wait=False and it is taken."""
    os.makedirs(f"{ROOT}/runs/.locks", exist_ok=True)
    fh = open(f"{ROOT}/runs/.locks/{name}.lock", "a")
    try: fcntl.flock(fh, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BlockingIOError: fh.close(); return None
    return fh


def scenes(): return {s["id"]: s for s in read_json(f"{ROOT}/scenes/index.json", [])}
def objects(scene): return read_json(f"{ROOT}/scenes/{scene}.json", {"scene": scene, "objects": []})


def find(run):
    """(scene entry, object entry) of a run."""
    for sid, sc in scenes().items():
        for o in objects(sid)["objects"]:
            if o["run"] == run: return sc, o
    sys.exit(f"[pipeline] unknown run {run!r}: not in any scenes/<scene>.json")


def edit_objects(scene, fn):
    """Change scenes/<scene>.json under its lock: fn(list of objects) changes the list in place."""
    fh = lock(f"objects_{scene}")
    try:
        doc = objects(scene); out = fn(doc["objects"]); write_json(f"{ROOT}/scenes/{scene}.json", doc); return out
    finally: fh.close()


def status(path, **kw):
    st = read_json(path, {}); st.update(kw, updated=now()); write_json(path, st)


# ---------------------------------------------------------------- object fields
def norm_cam(cam):
    """{C, Rb, f, size} as is; {yaw, pitch, fov[, size]} = the panorama camera turned that way (as common.yaw_pitch)."""
    if "yaw" in cam:
        y, p = math.radians(cam["yaw"]), math.radians(cam["pitch"]); S = int(cam.get("size", 1024))
        z = [math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)]
        n = math.hypot(z[2], z[0]); x = [z[2] / n, 0.0, -z[0] / n]                   # cross((0, 1, 0), z), normalised
        yv = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]   # cross(z, x)
        return dict(C=[0.0, 0.0, 0.0], Rb=[[x[i], yv[i], z[i]] for i in range(3)], f=S / 2 / math.tan(math.radians(cam["fov"]) / 2), size=S)
    C, Rb, f, S = cam["C"], cam["Rb"], float(cam["f"]), int(cam["size"])
    if len(C) != 3 or len(Rb) != 3 or any(len(r) != 3 for r in Rb) or f <= 0 or S <= 0: raise ValueError(f"bad cam {cam}")
    return dict(C=[float(x) for x in C], Rb=[[float(x) for x in r] for r in Rb], f=f, size=S)


def check_fields(o):
    """Validate / normalise the fields an object may be given (new and set)."""
    if "prompt" in o and not str(o["prompt"]).strip(): raise ValueError("prompt is empty")
    if "parts" in o: o["parts"] = [str(w) for w in o["parts"] if str(w).strip()]
    if "rest" in o and o["rest"] not in ("floor", "table"): raise ValueError(f"rest must be floor or table, not {o['rest']!r}")
    if "cover" in o: o["cover"] = bool(o["cover"])
    if "cam" in o: o["cam"] = norm_cam(o["cam"])
    if o.get("click") is not None: o["click"] = [float(o["click"][0]), float(o["click"][1])]
    return o


# ---------------------------------------------------------------- GPU: hold one card's lock for the whole run, as the sam3d pipeline does
def claim_gpu(need_mb, waiting):
    while True:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split("\n")
        cards = sorted((int(u), int(i), int(t)) for i, u, t in (r.replace(" ", "").split(",") for r in rows if r.strip()))
        for used, idx, total in cards:
            fh = open(f"/tmp/outdoor_pipeline_gpu{idx}.lock", "a")
            try: fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: fh.close(); continue
            if total - used >= need_mb: return idx, fh
            fh.close()
        waiting(); time.sleep(20)


# ---------------------------------------------------------------- running a module in its env
CHILD = [None]                                                  # the step process running now (for SIGTERM)


def call(module, ctx, env="sam3d", gpu=None):
    """Run gs2mesh.<module>.run(ctx) in its own process. Returns (exit code, last stderr lines)."""
    e = dict(os.environ, **CHILD_ENV)
    e["PATH"] = f"/usr/local/cuda-12.8/bin:{e.get('PATH', '')}"
    e["LD_LIBRARY_PATH"] = f"/usr/local/cuda-12.8/lib64:{e.get('LD_LIBRARY_PATH', '')}"
    e["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    p = subprocess.Popen([ENV[env], "-I", os.path.abspath(__file__), "_call", module, json.dumps(ctx)],
                         stderr=subprocess.PIPE, text=True, errors="replace", env=e)
    CHILD[0] = p; tail = collections.deque(maxlen=15)
    for line in p.stderr:
        sys.stderr.write(line); sys.stderr.flush()
        if line.strip(): tail.append(line.rstrip())
    p.wait(); CHILD[0] = None
    return p.returncode, "\n".join(tail)


def composable(o):
    d = f"{ROOT}/runs/{o['run']}"
    st = read_json(f"{d}/status.json", {}).get("state")
    return st not in ("queued", "running") and all(os.path.exists(f"{d}/{p}") for p in
                                                   ("labels.npy", "place.json", f"gen/{o['name']}_world.glb"))


def compose(scene):
    sc = scenes().get(scene) or sys.exit(f"[pipeline] unknown scene {scene!r}: not in scenes/index.json")
    out = f"{ROOT}/runs/{scene}_compose"; os.makedirs(out, exist_ok=True)
    fh = lock(f"compose_{scene}")
    try:
        objs = [o for o in objects(scene)["objects"] if composable(o)]
        status(f"{out}/status.json", state="running", error=None)
        log(f"compose {scene}: {[o['run'] for o in objs]}")
        ctx = dict(scene=scene, ply=f"{ROOT}/{sc['ply']}", out=out,
                   objects=[dict(dir=f"{ROOT}/runs/{o['run']}", run=o["run"], name=o["name"], label=o.get("label", "")) for o in objs])
        code, tail = call("compose", ctx)
        status(f"{out}/status.json", state="done" if code == 0 else "failed", error=None if code == 0 else tail)
        return code
    finally: fh.close()


# ---------------------------------------------------------------- commands
def cmd_steps(a):
    for i, (k, (env, mb, what)) in enumerate(STEPS.items(), 1):
        print(f"{i:2d}. {k:8s} [{env}, {'GPU ' + str(mb // 1000) + ' GB' if mb else 'CPU'}]  {what}")


def cmd_new(a):
    if a.scene not in scenes(): sys.exit(f"[pipeline] unknown scene {a.scene!r}: not in scenes/index.json")
    o = json.loads(a.json)
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", o.get("name", "")): sys.exit("[pipeline] name must be ascii letters, digits, _ or -")
    for k in ("prompt", "cam"):
        if k not in o: sys.exit(f"[pipeline] {k} is missing")
    o = check_fields({k: o[k] for k in ("name",) + FIELDS if k in o})
    o = dict(dict(label="", parts=[], rest="floor", cover=False, click=None), **o)

    def add(objs):
        taken = {x["run"] for x in objs}; base = f"{a.scene}_{o['name']}"; run, k = base, 1
        while run in taken or os.path.exists(f"{ROOT}/runs/{run}"): k += 1; run = f"{base}_{k}"
        objs.append(dict(run=run, **o)); return run
    print(json.dumps({"run": edit_objects(a.scene, add)}))


def cmd_set(a):
    sc, _ = find(a.run); new = check_fields({k: v for k, v in json.loads(a.json).items() if k in FIELDS})
    if "cam" in new and "click" not in new: new["click"] = None             # a click belongs to the old image

    def change(objs):
        for o in objs:
            if o["run"] == a.run: o.update(new)
    edit_objects(sc["id"], change)
    print(json.dumps({"run": a.run, "changed": sorted(new)}))


def cmd_run(a, steps=None):
    sc, o = find(a.run)
    if steps is None:
        names = list(STEPS)
        if a.steps: steps = [s.strip() for s in a.steps.split(",") if s.strip()]
        elif a.from_: steps = names[names.index(a.from_):] if a.from_ in STEPS else [a.from_]
        else: steps = names
        bad = [s for s in steps if s not in STEPS]
        if bad: sys.exit(f"[pipeline] unknown step(s) {bad}; steps: {' '.join(names)}")
        steps = [s for s in names if s in steps]                                  # always in pipeline order
    d = f"{ROOT}/runs/{a.run}"; os.makedirs(d, exist_ok=True); st = f"{d}/status.json"
    run_lock = lock(f"run_{a.run}", wait=False)
    if run_lock is None: sys.exit(f"[pipeline] 这个物体正在跑: {a.run} is already running")
    if getattr(a, "seed", None) is not None:
        def keep_seed(objs):
            for x in objs:
                if x["run"] == a.run: x["seed"] = a.seed
        edit_objects(sc["id"], keep_seed); o["seed"] = a.seed
    ctx = dict(o, run=a.run, dir=d, scene=sc["id"], ply=f"{ROOT}/{sc['ply']}", pano=f"{ROOT}/{sc['pano']}", seed=o.get("seed", 42))
    state = dict(step=None, i=0, n=len(steps))

    def stop(signum, frame):
        if CHILD[0] is not None:
            CHILD[0].terminate()
            try: CHILD[0].wait(10)
            except subprocess.TimeoutExpired: CHILD[0].kill()
        status(st, state="failed", error="已停止", **state); log("stopped"); os._exit(143)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)

    status(st, state="queued", steps=steps, error=None, gpu=None, pid=os.getpid(), **state)
    need = max(STEPS[s][1] for s in steps); gpu, gpu_lock = None, None
    if need:
        gpu, gpu_lock = claim_gpu(need, lambda: log(f"waiting for a GPU with {need} MB free"))
    status(st, state="running", gpu=gpu)
    log(f"{a.run} on {'GPU ' + str(gpu) if gpu is not None else 'CPU'}: {' '.join(steps)}")
    for i, s in enumerate(steps):
        state.update(step=s, i=i); status(st, **state)
        log(f"== {s} ({i + 1}/{len(steps)})"); t0 = time.time()
        code, tail = call(s, ctx, STEPS[s][0], gpu)
        if code != 0:
            status(st, state="failed", error=tail or f"exit code {code}", **state)
            log(f"{s} failed after {time.time() - t0:.0f} s"); sys.exit(1)
        state.update(i=i + 1); status(st, **state); log(f"{s} done in {time.time() - t0:.0f} s")
    status(st, state="done", **state)
    if gpu_lock: gpu_lock.close()
    run_lock.close()
    if RECOMPOSE & set(steps) and composable(o): sys.exit(compose(sc["id"]))


def cmd_apply(a): cmd_run(a, ["carve", "compare", "debug"])
def cmd_compose(a): sys.exit(compose(a.scene))


def cmd_remove(a):
    sc, _ = find(a.run)
    edit_objects(sc["id"], lambda objs: objs.__setitem__(slice(None), [x for x in objs if x["run"] != a.run]))
    log(f"{a.run} taken out of {sc['id']} (runs/{a.run} stays)")
    sys.exit(compose(sc["id"]))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_call":                       # inside a step's process
        import importlib, types
        sys.path.insert(0, ROOT)
        importlib.import_module(f"gs2mesh.{sys.argv[2]}").run(types.SimpleNamespace(**json.loads(sys.argv[3])))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("steps").set_defaults(fn=cmd_steps)
    p = sub.add_parser("new"); p.add_argument("scene"); p.add_argument("json"); p.set_defaults(fn=cmd_new)
    p = sub.add_parser("set"); p.add_argument("run"); p.add_argument("json"); p.set_defaults(fn=cmd_set)
    p = sub.add_parser("run"); p.add_argument("run"); g = p.add_mutually_exclusive_group()
    g.add_argument("--steps"); g.add_argument("--from", dest="from_"); p.add_argument("--seed", type=int); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("apply"); p.add_argument("run"); p.set_defaults(fn=cmd_apply)
    p = sub.add_parser("compose"); p.add_argument("scene"); p.set_defaults(fn=cmd_compose)
    p = sub.add_parser("remove"); p.add_argument("run"); p.set_defaults(fn=cmd_remove)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
