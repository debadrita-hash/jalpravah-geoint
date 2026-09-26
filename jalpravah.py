"""
JalPravah: local web application for the hydrosim dam-break framework.

    python jalpravah.py [--port 8765] [--no-browser]

Pages
    /                           scenarios (home)
    /builder                    create a scenario anywhere (open DEM, imagery and OSM dam data) and run the models
    /satellite                  near-real-time flood mapping with Sentinel-1 on Google Earth Engine
    /run/<scenario>/...         overview, SPH, Delft3D, 3-D views and comparison of one scenario
API
    /api/scenarios              scenarios with live solver status
    /api/jobs                   background jobs (build, Delft3D, SPH, exposure, satellite) and their logs
    /api/geocode, /api/dams     place search (Nominatim) and dams from OpenStreetMap
    /api/gee/*                  Earth Engine connection and flood-mapping runs
    /run/<s>/api/status|manifest, /run/<s>/files/<path>
"""
import argparse
import datetime as dt
import hashlib
import hmac
import http.server
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

from hydrosim.config import RUNS_ROOT, Scenario

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "hydrosim" / "viewer_assets"
JOBS_DIR = RUNS_ROOT / "_jobs"
SAT_DIR = RUNS_ROOT / "_satellite"
REMOTE_FILE = HERE / ".jalpravah_remote.json"  # written by jalpravah_online.py; never committed
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/geo+json", ".geojson")
mimetypes.add_type("application/vnd.google-earth.kml+xml", ".kml")


def docker(args, timeout=20):
    try:
        r = subprocess.run(["docker"] + args, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, MSYS_NO_PATHCONV="1"))
        return r.returncode, r.stdout
    except Exception:
        return 1, ""


def scenarios():
    out = {}
    for cfg in sorted((HERE / "config").glob("*.json")):
        if cfg.name.startswith("gee_"):
            continue
        try:
            sc = Scenario(cfg)
        except Exception:
            continue
        if (sc.viewer_dir / "index.html").exists():
            out[sc.name] = sc
    return out


def _bbox(sc, meta):
    """[lon_min, lat_min, lon_max, lat_max] of the scenario corridor."""
    if not meta:
        return None
    from hydrosim.exposure import corridor_bbox_wgs84
    s, w, n, e = corridor_bbox_wgs84(meta)
    return [w, s, e, n]


def remote_key():
    try:
        return json.loads(REMOTE_FILE.read_text()).get("key", "")
    except Exception:
        return ""


def download_token(key, path):
    return hmac.new(key.encode(), path.encode(), hashlib.sha256).hexdigest()[:32]


def zip_shapefile(shp):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            f = shp.with_suffix(ext)
            if f.exists():
                z.write(f, f.name)
    return buf.getvalue()


class Status:
    """Solver progress per scenario, cached 20 s so page polling never hammers Docker."""

    def __init__(self):
        self.cache, self.lock = {}, threading.Lock()

    def get(self, sc):
        with self.lock:
            t, d = self.cache.get(sc.name, (0, None))
            if time.time() - t > 20:
                d = {"sph": self._sph(sc), "delft3d": self._d3d(sc), "checked": dt.datetime.now().isoformat(timespec="seconds")}
                self.cache[sc.name] = (time.time(), d)
            return d

    @staticmethod
    def _sph(sc):
        planned = float(sc["sph"]["duration_s"])
        vol = f"hydrosim_{sc.name}_sph"
        _, vols = docker(["volume", "ls", "-q"])
        if vol not in vols.split():
            return {"running": False, "simulated_s": 0.0, "planned_s": planned, "eta": None, "state": "not run"}
        _, running = docker(["ps", "-q", "-f", "name=hydrosim_sph_solver", "-f", f"volume={vol}"])
        _, out = docker(["run", "--rm", "-v", f"{vol}:/work:ro", "alpine", "sh", "-c",
                         "d=$(cd /work/case && ls -d out out_ext* 2>/dev/null | sort -V | tail -1); grep -E '^[0-9]{5} ' /work/case/$d/Run.out | tail -1"])
        sim, eta = 0.0, None
        m = re.match(r"\d+\s+([\d.]+)\s+.*\s(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})\s*$", out.strip())
        if m:
            sim = float(m.group(1))
            eta = dt.datetime.strptime(m.group(2), "%d-%m-%Y %H:%M:%S").replace(tzinfo=dt.timezone.utc).astimezone().strftime("%d %b %H:%M")
        live = bool(running.strip())
        done = (sc.results_dir / "sph" / "metrics.json").exists() and not live and sim >= planned - 1
        return {"running": live, "simulated_s": sim, "planned_s": planned, "eta": eta if live else None,
                "state": "running" if live else ("complete" if done else ("stopped" if sim else "not run"))}

    @staticmethod
    def _d3d(sc):
        planned = float(sc["simulation"]["duration_s"])
        _, running = docker(["ps", "-q", "-f", "name=hydrosim_d3d_solver", "-f", f"volume=hydrosim_{sc.name}_d3d"])
        live = bool(running.strip())
        log = sc.delft3d_dir / "case" / "run.log"
        tail = log.read_text(errors="ignore")[-3000:] if log.exists() else ""
        metrics = sc.results_dir / "delft3d" / "metrics.json"
        if live:
            pct = re.findall(r"([\d.]+)% completed", tail)
            return {"running": True, "simulated_s": planned * float(pct[-1]) / 100 if pct else 0.0, "planned_s": planned,
                    "eta": None, "state": "running"}
        ok = "shutting down normally" in tail
        if ok and metrics.exists() and metrics.stat().st_mtime >= log.stat().st_mtime:
            return {"running": False, "simulated_s": planned, "planned_s": planned, "eta": None, "state": "complete"}
        if ok:
            return {"running": False, "simulated_s": planned, "planned_s": planned, "eta": None, "state": "analysing"}
        return {"running": False, "simulated_s": 0.0, "planned_s": planned, "eta": None,
                "state": "failed" if "exited abnormally" in tail else "not run"}


def _alive(pid):
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


class Jobs:
    """Background jobs started from the web app; each writes a log in runs/_jobs."""

    def __init__(self):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.file = JOBS_DIR / "jobs.json"
        self.jobs = json.loads(self.file.read_text()) if self.file.exists() else {}
        self.procs = {}

    def _save(self):
        self.file.write_text(json.dumps(self.jobs, indent=1))

    def _launch(self, jid, cmd):
        log = JOBS_DIR / f"{jid}.log"
        fh = open(log, "w", encoding="utf-8")
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0
        p = subprocess.Popen(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                             creationflags=flags, start_new_session=(os.name != "nt"))  # survives a server restart
        self.procs[jid] = (p, fh)
        j = self.jobs[jid]
        j.update(state="running", started=dt.datetime.now().isoformat(timespec="seconds"), log=str(log), pid=p.pid)
        j.pop("cmd", None)

    def remove(self, jid):
        """Deletes a finished/failed job (log + input file) or cancels a queued one. Running jobs are kept."""
        with self.lock:
            j = self.jobs.get(jid)
            if not j:
                return False, "Job not found"
            if j["state"] == "running" and (jid in self.procs or _alive(j.get("pid"))):
                return False, "This job is still running; its log is kept until it finishes."
            for f in (j.get("log"), j.get("params")):
                if f:
                    Path(f).unlink(missing_ok=True)
            del self.jobs[jid]
            self._save()
            return True, "cancelled" if j["state"] == "queued" else "deleted"

    def forget_scenario(self, name):
        for jid in [k for k, j in self.jobs.items() if j.get("scenario") == name and j["state"] != "running"]:
            self.remove(jid)

    def start(self, kind, title, cmd, scenario=None, queue_if_busy=False, params=None):
        """Starts a job; with queue_if_busy a job of the same kind already running makes this one wait its turn."""
        with self.lock:
            jid = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f_") + kind  # microseconds: unique even for rapid clicks
            while jid in self.jobs:
                jid += "_"
            self.jobs[jid] = {"id": jid, "kind": kind, "title": title, "scenario": scenario, "state": "queued", "params": params,
                              "queued": dt.datetime.now().isoformat(timespec="seconds"), "started": None, "log": None, "cmd": cmd}
            busy = any(j["kind"] == kind and j["state"] == "running" for j in self.jobs.values())
            if not (queue_if_busy and busy):
                self._launch(jid, cmd)
            self._save()
            return {k: v for k, v in self.jobs[jid].items() if k != "cmd"}

    def refresh(self):
        with self.lock:
            for jid, (p, fh) in list(self.procs.items()):
                rc = p.poll()
                if rc is not None:
                    fh.close()
                    self.jobs[jid]["state"] = "finished" if rc == 0 else "failed"
                    self.jobs[jid]["ended"] = dt.datetime.now().isoformat(timespec="seconds")
                    del self.procs[jid]
            for j in self.jobs.values():  # jobs started by an earlier server session: follow them by process id
                if j["state"] == "running" and j["id"] not in self.procs:
                    if not _alive(j.get("pid")):
                        tail = Path(j["log"]).read_text(errors="ignore")[-3000:] if j.get("log") and Path(j["log"]).exists() else ""
                        j["state"] = "failed" if "Traceback" in tail else "finished"
                        j["ended"] = dt.datetime.now().isoformat(timespec="seconds")
            # start the oldest queued job of a kind once nothing of that kind is running
            for jid in sorted(k for k, j in self.jobs.items() if j["state"] == "queued" and j.get("cmd")):
                kind = self.jobs[jid]["kind"]
                if not any(j["kind"] == kind and j["state"] == "running" for j in self.jobs.values()):
                    self._launch(jid, self.jobs[jid]["cmd"])
            self._save()
            return sorted(({k: v for k, v in j.items() if k != "cmd"} for j in self.jobs.values()), key=lambda j: j["id"], reverse=True)

    def active(self, kind=None, scenario=None):
        return [j for j in self.refresh() if j["state"] == "running" and (kind is None or j["kind"] == kind)
                and (scenario is None or j.get("scenario") == scenario)]

    def log(self, jid, tail=6000):
        j = self.jobs.get(jid)
        if not j or not j.get("log"):
            return ""
        t = Path(j["log"]).read_text(errors="ignore") if Path(j["log"]).exists() else ""
        return t[-tail:]


def make_handler(status, jobs):
    py = sys.executable

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        # ------------------------------------------------------------ helpers
        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            return self._send(json.dumps(obj).encode(), "application/json", code)

        def _file(self, f, download=False):
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(f.stat().st_size))
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{f.name}"')
            self.end_headers()
            with open(f, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    self.wfile.write(chunk)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _remote_ok(self):
            """Requests from this computer are always served. Requests that arrive through the Cloudflare tunnel
            (JalPravah Online) must carry the shared key from the Vercel site, or a signed download token."""
            if "Cf-Connecting-Ip" not in self.headers and "Cf-Ray" not in self.headers:
                return True
            key = remote_key()
            if key and hmac.compare_digest(self.headers.get("X-JalPravah-Key", ""), key):
                return True
            u = urllib.parse.urlparse(self.path)
            tok = urllib.parse.parse_qs(u.query).get("jpdl", [""])[0]
            if key and self.command == "GET" and tok and hmac.compare_digest(tok, download_token(key, urllib.parse.unquote(u.path))):
                return True
            self.send_error(403, "Forbidden")
            return False

        # ------------------------------------------------------------ GET
        def do_GET(self):
            if not self._remote_ok():
                return
            u = urllib.parse.urlparse(self.path)
            path, qs = urllib.parse.unquote(u.path), urllib.parse.parse_qs(u.query)
            page = {"/": "home.html", "/index.html": "home.html", "/builder": "builder.html", "/satellite": "satellite.html"}.get(path)
            if page:
                return self._file(ASSETS / page)
            if path in ("/hydro.css", "/hydro.js", "/logo.png"):
                return self._file(ASSETS / path.lstrip("/"))
            if path == "/api/scenarios":
                out = []
                for name, sc in scenarios().items():
                    meta = sc.load_meta() if sc.meta_file.exists() else {}
                    out.append({"name": name, "title": sc.cfg.get("title", name.replace("_", " ")), "description": sc["description"], "reach_km": sc["domain"]["reach_length_km"],
                                "dam": {k: meta.get("dam", {}).get(k) for k in ("height_m", "crest_m")},
                                "lat": sc["dam"]["lat"], "lon": sc["dam"]["lon"], "type": sc["dam"].get("type", "dam"),
                                "reservoir_Mm3": round(meta.get("reservoir", {}).get("volume_m3", 0) / 1e6, 2),
                                "bbox": _bbox(sc, meta),
                                "extents": {m: f"/run/{name}/files/results/{m}/gis/{m}_flood_extent.geojson"
                                            for m in ("delft3d", "sph") if (sc.results_dir / m / "gis" / f"{m}_flood_extent.geojson").exists()},
                                "status": status.get(sc)})
                return self._json(out)
            if path == "/api/jobs":
                return self._json(jobs.refresh())
            if path == "/api/scenarios/size":
                scs = scenarios()
                name = qs.get("name", [""])[0]
                if name not in scs:
                    return self._json({"error": "Unknown scenario"}, 404)
                total = sum(f.stat().st_size for f in scs[name].run_dir.rglob("*") if f.is_file())
                return self._json({"bytes": total})
            m = re.match(r"^/api/jobs/([^/]+)/log$", path)
            if m:
                return self._send(jobs.log(m.group(1)).encode(), "text/plain; charset=utf-8")
            if path == "/api/geocode":
                from hydrosim import opendata
                try:
                    return self._json(opendata.geocode(qs.get("q", [""])[0]))
                except Exception as ex:
                    return self._json({"error": f"Place search failed: {ex}"}, 502)
            if path == "/api/dams":
                from hydrosim import opendata
                try:
                    bbox = [float(v) for v in qs["bbox"][0].split(",")]
                    return self._json(opendata.find_dams(bbox))
                except Exception as ex:
                    return self._json({"error": f"OpenStreetMap dam search failed: {ex}"}, 502)
            if path == "/api/gee/status":
                from hydrosim import gee
                return self._json(gee.status())
            if path == "/api/gee/runs":
                runs = []
                for d in sorted(SAT_DIR.glob("*/info.json"), reverse=True):
                    runs.append({"id": d.parent.name, **json.loads(d.read_text())})
                return self._json(runs)
            m = re.match(r"^/satellite-files/([^/]+)/([^/]+)$", path)
            if m and m.group(2).endswith(".shp.zip"):
                shp = (SAT_DIR / m.group(1) / m.group(2)[:-4]).resolve()
                if SAT_DIR.resolve() not in shp.parents or not shp.exists():
                    return self.send_error(404)
                self.send_response(200)
                data = zip_shapefile(shp)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{shp.stem}_{m.group(1)}.zip"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if m:
                f = (SAT_DIR / m.group(1) / m.group(2)).resolve()
                if SAT_DIR.resolve() not in f.parents or not f.is_file():
                    return self.send_error(404)
                return self._file(f, download=not f.name.endswith(".geojson"))
            m = re.match(r"^/run/([^/]+)(/.*)?$", path)
            if not m:
                return self.send_error(404, "Not found")
            scs = scenarios()
            if m.group(1) not in scs:
                return self.send_error(404, "Unknown scenario")
            sc, rest = scs[m.group(1)], (m.group(2) or "/")
            if rest == "/":
                rest = "/index.html"
            if rest == "/api/status":
                return self._json(status.get(sc))
            if rest == "/api/manifest":
                from hydrosim.archive import write_manifest
                return self._json(write_manifest(sc))
            base, download = (sc.run_dir.resolve(), True) if rest.startswith("/files/") else (sc.viewer_dir.resolve(), False)
            rel = rest[len("/files/"):] if download else rest.lstrip("/")
            if download and rel.endswith(".shp.zip"):
                shp = (base / rel[:-4]).resolve()
                if base not in shp.parents or not shp.exists():
                    return self.send_error(404, "File not found")
                data = zip_shapefile(shp)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{sc.name}_{shp.stem}.zip"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            f = (base / rel).resolve()
            if base not in f.parents or not f.is_file():
                return self.send_error(404, "File not found")
            return self._file(f, download and not f.name.endswith(".geojson"))

        # ------------------------------------------------------------ POST
        def do_POST(self):
            if not self._remote_ok():
                return
            path = urllib.parse.urlparse(self.path).path
            try:
                body = self._body()
            except Exception:
                return self._json({"error": "Request body must be JSON"}, 400)
            if path == "/api/scenarios":
                need = ("name", "lat", "lon", "height_m")
                if any(str(body.get(k, "")).strip() == "" for k in need):
                    return self._json({"error": "Name, dam location and dam height are required."}, 400)
                pf = JOBS_DIR / f"params_{int(time.time() * 1000)}.json"
                pf.write_text(json.dumps(body))
                from hydrosim.builder import slug
                j = jobs.start("build", f"Build scenario: {body['name']}", [py, "-m", "hydrosim.builder", str(pf)],
                               scenario=slug(body["name"]), queue_if_busy=True, params=str(pf))
                return self._json(j)
            if path == "/api/run":
                scs = scenarios()
                name, model = body.get("scenario"), body.get("model")
                if name not in scs or model not in ("delft3d", "sph"):
                    return self._json({"error": "Unknown scenario or model"}, 400)
                sc = scs[name]
                st = status.get(sc)[model]
                if st["running"] or jobs.active(model, name):
                    return self._json({"error": f"{model} is already running for this scenario."}, 409)
                if model == "sph":
                    _, other = docker(["ps", "-q", "-f", "name=hydrosim_sph_solver"])
                    if other.strip() or jobs.active("sph"):
                        return self._json({"error": "An SPH simulation is already using the processor. Start this one when it finishes."}, 409)
                    cfg = str(sc.path)
                    cmd = [py, "-c", "import subprocess,sys; r=subprocess.call([sys.executable,'run_sph.py',sys.argv[1]]); "
                                     "sys.exit(r or subprocess.call([sys.executable,'run_sph_extend.py',sys.argv[1]]))", cfg]
                else:
                    cmd = [py, "run_delft3d.py", str(sc.path), "--analyze", "--cpus", body.get("cpus", "10-11")]
                return self._json(jobs.start(model, f"{'SPH' if model == 'sph' else 'Delft3D'}: {name}", cmd, name))
            if path == "/api/jobs/delete":
                ok, msg = jobs.remove(body.get("id", ""))
                return self._json({"ok": ok, "message": msg}, 200 if ok else 409)
            if path == "/api/scenarios/delete":
                scs = scenarios()
                name = body.get("scenario")
                if name not in scs:
                    return self._json({"error": "Unknown scenario"}, 404)
                sc = scs[name]
                st = status.get(sc)
                _, live = docker(["ps", "-q", "-f", f"volume=hydrosim_{name}_sph", "-f", f"volume=hydrosim_{name}_d3d"])
                _, live2 = docker(["ps", "-q", "-f", f"volume=hydrosim_{name}_d3d"])
                if st["sph"]["running"] or st["delft3d"]["running"] or live.strip() or live2.strip() or jobs.active(scenario=name):
                    return self._json({"error": "A simulation or build for this scenario is still running. Wait for it to finish before deleting."}, 409)
                import shutil
                shutil.rmtree(sc.run_dir, ignore_errors=True)
                sc.path.unlink(missing_ok=True)
                docker(["volume", "rm", "-f", f"hydrosim_{name}_sph", f"hydrosim_{name}_d3d"], timeout=120)
                jobs.forget_scenario(name)
                status.cache.pop(name, None)
                return self._json({"ok": True, "message": f"{sc.cfg.get('title', name)} deleted"})
            if path == "/api/exposure":
                scs = scenarios()
                if body.get("scenario") not in scs:
                    return self._json({"error": "Unknown scenario"}, 400)
                sc = scs[body["scenario"]]
                code = ("import sys, pipeline; from hydrosim.config import Scenario; sc=Scenario(sys.argv[1]); "
                        "from hydrosim.results import ResultStore; "
                        "[pipeline.refresh_metrics(sc,m) for m in ('delft3d','sph') if (ResultStore(sc,m).dir/'frames.npz').exists()]; "
                        "pipeline.step_viewer(sc)")
                (sc.scenario_dir / "osm_exposure.json").unlink(missing_ok=True)
                return self._json(jobs.start("exposure", f"Exposure and damage: {sc.name}", [py, "-c", code, str(sc.path)], sc.name))
            if path == "/api/gee/auth":
                from hydrosim import gee
                gee.authenticate()
                return self._json({"message": "A Google sign-in page has opened in your browser. Finish it, then return here."})
            if path == "/api/gee/project":
                from hydrosim import gee
                gee.save_project(body.get("project", ""))
                return self._json(gee.status())
            if path == "/api/gee/run":
                try:
                    bbox = [float(v) for v in body["bbox"]]
                    pre, post = body["pre"], body["post"]
                except Exception:
                    return self._json({"error": "Area and both date windows are required."}, 400)
                if jobs.active("satellite"):
                    return self._json({"error": "A satellite analysis is already running."}, 409)
                run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                pf = JOBS_DIR / f"gee_{run_id}.json"
                pf.write_text(json.dumps({"bbox": bbox, "pre": pre, "post": post, "threshold_db": body.get("threshold_db", 3.0),
                                          "orbit": body.get("orbit", "DESCENDING"), "label": body.get("label", ""), "run_id": run_id}))
                return self._json(jobs.start("satellite", f"Sentinel-1 flood map {run_id}", [py, "-m", "hydrosim.gee", str(pf)]))
            return self._json({"error": "Not found"}, 404)

    return Handler


def main():
    ap = argparse.ArgumentParser(description="JalPravah local web application")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(Status(), Jobs()))
    url = f"http://localhost:{a.port}/"
    print(f"JalPravah is running at {url}  (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
