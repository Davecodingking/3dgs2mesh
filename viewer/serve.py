# Serves 3dgs2mesh/ to the tool page (viewer/index.html) on 127.0.0.1 and turns its buttons into pipeline.py calls.
# Long jobs start in the background (own session, output appended to runs/<run>/run.log) and answer at once; the page
# follows them in runs/<run>/status.json. Only the add-object probe (view + mask, tens of seconds) waits.
#   POST /api/object?scene=<id>[&run=<run>]  body: prompt, name, label, parts, rest, cover, cam, click
#        new <scene> (or set <run> to redo an object's probe), then run --steps view,mask; answers {run}
#   POST /api/run?run=<run>[&from=<step> | &steps=a,b][&seed=N]   run (part of) the pipeline
#   POST /api/stop?run=<run>       SIGTERM to the run's pipeline process (it stops its step and writes "已停止")
#   POST /api/remove?run=<run>     take the object off its scene's list (the scene is recomposed)
#   POST /api/manual?run=<run>     body {"delete": [scene indices], "keep": [...]} -> runs/<run>/manual.json
#   POST /api/apply?run=<run>      redo the deletion with the hand edits, the comparisons, the check data, the scene
#   POST /api/compose?scene=<id>   recompose the scene
# usage: python viewer/serve.py [port=8791]
import json, os, re, signal, sys, subprocess, http.server, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
NAME, STEPS = re.compile(r"[A-Za-z0-9_\-]+"), re.compile(r"[a-z]+(,[a-z]+)*")
jobs = []                                                    # background processes, reaped as new ones start


def pipeline(*args):
    return [PY, f"{ROOT}/pipeline.py", *args]


def background(cmd, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    jobs[:] = [p for p in jobs if p.poll() is None]
    with open(log, "a") as f:
        jobs.append(subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))


def status(run):
    try: return json.load(open(f"{ROOT}/runs/{run}/status.json"))
    except (OSError, ValueError): return {}


def running_pid(run):
    """The pid of <run>'s pipeline if it is really still running (a pid in an old status.json may belong to anyone now)."""
    st = status(run)
    if st.get("state") not in ("queued", "running") or not st.get("pid"): return None
    try: cmd = open(f"/proc/{int(st['pid'])}/cmdline", "rb").read().split(b"\0")
    except OSError: return None
    return int(st["pid"]) if b"pipeline.py" in b" ".join(cmd) and run.encode() in cmd else None


def why(p, run=None):
    """What went wrong, short: the error the pipeline wrote into status.json, else the last lines of stderr."""
    st = status(run) if run else {}
    if st.get("state") == "failed" and st.get("error"): return st["error"]
    return "\n".join([l for l in (p.stderr or "").strip().splitlines() if l.strip()][-3:]) or f"exit code {p.returncode}"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):                                   # always fresh data after a job
        self.send_header("Cache-Control", "no-store"); super().end_headers()

    def reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        run, scene = q.get("run", ""), q.get("scene", "")
        if any(v and not NAME.fullmatch(v) for v in (run, scene)): return self.reply(400, {"error": "bad run / scene name"})
        if u.path == "/api/object": return self.probe(scene, run)
        if u.path == "/api/compose":
            if not os.path.exists(f"{ROOT}/scenes/{scene}.json"): return self.reply(400, {"error": f"unknown scene {scene!r}"})
            background(pipeline("compose", scene), f"{ROOT}/runs/{scene}_compose/compose.log")
            return self.reply(200, {"started": True})
        rd = f"{ROOT}/runs/{run}"
        if not run or not os.path.isdir(rd): return self.reply(400, {"error": f"unknown run {run!r}"})
        if u.path == "/api/manual":
            body = self.body()
            man = {k: sorted({int(i) for i in body.get(k, [])}) for k in ("delete", "keep")}
            json.dump(man, open(f"{rd}/manual.json", "w"))
            return self.reply(200, {"saved": {k: len(v) for k, v in man.items()}})
        if u.path == "/api/stop":
            pid = running_pid(run)
            if not pid: return self.reply(409, {"error": "这个物体没在跑"})
            os.kill(pid, signal.SIGTERM)
            return self.reply(200, {"stopped": pid})
        if u.path in ("/api/run", "/api/apply", "/api/remove"):
            if running_pid(run): return self.reply(409, {"error": "这个物体正在跑，先等它跑完或者停掉"})
            if u.path == "/api/run":
                args = ["run", run]
                for key, flag in (("steps", "--steps"), ("from", "--from")):
                    if key in q:
                        if not STEPS.fullmatch(q[key]): return self.reply(400, {"error": f"bad {key}"})
                        args += [flag, q[key]]; break
                if "seed" in q: args += ["--seed", str(int(q["seed"]))]
            else:
                args = [u.path.rsplit("/", 1)[1], run]
            background(pipeline(*args), f"{rd}/run.log")
            return self.reply(200, {"started": True})
        self.reply(404, {"error": "unknown endpoint"})

    def probe(self, scene, run):
        """A new object (or a redo of one that has not been generated yet): its fields into the scene's list, then the
        generation image and the mask at the click. Waits for both and answers {run}, or {error, run}."""
        fields = self.body()
        if run:
            if not os.path.isdir(f"{ROOT}/runs/{run}"): return self.reply(400, {"error": f"unknown run {run!r}"})
            if running_pid(run): return self.reply(409, {"error": "这个物体正在跑，先等它跑完或者停掉", "run": run})
            fields.pop("name", None)
            p = subprocess.run(pipeline("set", run, json.dumps(fields, ensure_ascii=False)), cwd=ROOT, capture_output=True, text=True)
            if p.returncode: return self.reply(500, {"error": why(p), "run": run})
        else:
            if not os.path.exists(f"{ROOT}/scenes/{scene}.json"): return self.reply(400, {"error": f"unknown scene {scene!r}"})
            p = subprocess.run(pipeline("new", scene, json.dumps(fields, ensure_ascii=False)), cwd=ROOT, capture_output=True, text=True)
            if p.returncode: return self.reply(500, {"error": why(p)})
            run = json.loads(p.stdout.strip().splitlines()[-1])["run"]
        with open(f"{ROOT}/runs/{run}/run.log", "a") as log:
            p = subprocess.run(pipeline("run", run, "--steps", "view,mask"), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.PIPE, text=True)
            log.write(p.stderr)
        if p.returncode: return self.reply(500, {"error": why(p, run), "run": run})
        return self.reply(200, {"run": run})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    print(f"[serve] {ROOT} on http://127.0.0.1:{port}/viewer/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
