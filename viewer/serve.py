# Serves 3dgs2mesh/ to the viewer pages (index.html, debug.html) on 127.0.0.1 and takes the 3D check page's hand edits:
#   POST /api/manual?run=<run>   body {"delete": [scene indices], "keep": [...]}  -> runs/<run>/manual.json
#   POST /api/apply?run=<run>    re-runs carve_labels.py, view_compare.py and debug_export.py for that run, then compose.py
#                                for every composed scene (runs/*/objects.json) that contains it
#   POST /api/mesh_erase?run=<run>  body {"scale": [..3], "offset": [..3]}: mesh_erase.py (the scaled mesh as a 3D brush
#                                adds what it covers to manual.json), then the same as apply
# usage: python viewer/serve.py [port=8791]
import json, os, re, sys, glob, subprocess, http.server, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):                                   # always fresh data after an apply
        self.send_header("Cache-Control", "no-store"); super().end_headers()

    def reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); run = urllib.parse.parse_qs(u.query).get("run", [""])[0]
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", run) or not os.path.isdir(f"{ROOT}/runs/{run}"):
            return self.reply(400, {"error": f"unknown run {run!r}"})
        rd = f"{ROOT}/runs/{run}"
        if u.path == "/api/manual":
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            man = {k: sorted({int(i) for i in body.get(k, [])}) for k in ("delete", "keep")}
            json.dump(man, open(f"{rd}/manual.json", "w"))
            return self.reply(200, {"saved": {k: len(v) for k, v in man.items()}})
        if u.path in ("/api/apply", "/api/mesh_erase"):
            meta = json.load(open(f"{rd}/debug/meta.json")); scene, name = meta["scene_ply"], meta["name"]
            cmds = [[PY, "-I", f"{ROOT}/scripts/carve_labels.py", rd, scene, f"{rd}/scene_carved.ply"],
                    [PY, "-I", f"{ROOT}/scripts/view_compare.py", rd, scene, f"{rd}/scene_carved.ply"],
                    [PY, "-I", f"{ROOT}/scripts/debug_export.py", rd, scene, name]]
            if u.path == "/api/mesh_erase":
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                prm = dict(scale=[float(x) for x in body["scale"]], offset=[float(x) for x in body["offset"]])
                cmds.insert(0, [PY, "-I", f"{ROOT}/scripts/mesh_erase.py", rd, json.dumps(prm)])
            for oj in glob.glob(f"{ROOT}/runs/*/objects.json"):
                runs = [o["run"] for o in json.load(open(oj))["objects"]]
                if run in runs:
                    cmds.append([PY, "-I", f"{ROOT}/scripts/compose.py", scene, os.path.dirname(oj)] + [f"{ROOT}/runs/{r}" for r in runs])
            log = []
            for c in cmds:
                p = subprocess.run(c, cwd=ROOT, capture_output=True, text=True)
                log.append(" ".join(os.path.basename(x) for x in c[2:3]) + ": " + (p.stdout.strip().splitlines() or [""])[-1])
                if p.returncode != 0:
                    return self.reply(500, {"error": p.stderr.strip()[-800:], "log": log})
            return self.reply(200, {"log": log})
        self.reply(404, {"error": "unknown endpoint"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    print(f"[serve] {ROOT} on http://127.0.0.1:{port}/viewer/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
