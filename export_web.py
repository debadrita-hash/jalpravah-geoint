"""
JalPravah for web hosting (Vercel), written to web/.

    python export_web.py [--out web]

The site is the JalPravah app itself (same pages as http://localhost:8765/) plus:
  * a Vercel function (api/proxy.js) that forwards live requests through a Cloudflare tunnel to the JalPravah
    workstation started with jalpravah_online.py, so New scenario, Run, Delete and Satellite flood mapping work
    as they do locally; actions need the JalPravah password, reading is public
  * a workstation status / sign-in bar (jp-remote.js)
  * the finished scenarios as static files, and a snapshot of their status and file lists, so results stay
    viewable (and large 3-D data loads fast) while the workstation is offline
Only scenarios with at least one finished model are copied as static files; the others are served live.
"""
import argparse
import json
import shutil
from pathlib import Path

import jalpravah
from hydrosim.archive import write_manifest

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "hydrosim" / "viewer_assets"
REMOTE = HERE / "hydrosim" / "web_remote"

# result files published per model (the multi-GB solver cases and raw output are served live from the workstation)
MODEL_FILES = ["metrics.json", "impacts.json", "series.json"]
GIS_EXT = {".tif", ".geojson", ".kml", ".shp", ".shx", ".dbf", ".prj", ".cpg"}
SCENARIO_GIS_EXT = {".geojson", ".kml", ".shp", ".shx", ".dbf", ".prj", ".cpg"}
REMOTE_TAG = '<script src="/jp-remote.js" defer></script>\n'


def replace_once(text, old, new, where):
    if old not in text:
        raise SystemExit(f"export_web: expected text not found in {where}: {old[:60]!r}")
    return text.replace(old, new, 1)


def export(out):
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():  # keep dot files (.gitignore, a linked .vercel project)
        if not old.name.startswith("."):
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    for f in ("hydro.css", "hydro.js", "logo.png"):
        shutil.copy2(ASSETS / f, out / f)
    shutil.copy2(REMOTE / "jp-remote.js", out / "jp-remote.js")
    (out / "api").mkdir()
    shutil.copy2(REMOTE / "proxy.js", out / "api" / "proxy.js")
    snap = out / "snapshot"

    status = jalpravah.Status()
    listing = []
    for name, sc in jalpravah.scenarios().items():
        st = status.get(sc)
        done = [m for m in ("delft3d", "sph") if st[m]["state"] == "complete"]
        if not done:
            print(f"[web] {name}: no finished model run, served live only")
            continue
        run = out / "run" / name
        shutil.copytree(sc.viewer_dir, run)

        # downloadable files: model results + GIS, comparison, scenario geometry
        published = []
        for m in done:
            rdir = sc.results_dir / m
            published += [rdir / f for f in MODEL_FILES if (rdir / f).exists()]
            published += [f for f in sorted((rdir / "gis").glob("*")) if f.suffix in GIS_EXT]
        published += [f for f in [sc.results_dir / "comparison.json"] if f.exists()]
        published += [f for f in sorted((sc.scenario_dir / "gis").glob("*")) if f.suffix in SCENARIO_GIS_EXT]
        for f in published:
            dst = run / "files" / f.relative_to(sc.run_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        for shp in [f for f in published if f.suffix == ".shp"]:
            dst = run / "files" / shp.relative_to(sc.run_dir)
            dst.with_name(dst.name + ".zip").write_bytes(jalpravah.zip_shapefile(shp))

        # offline snapshot: status and the files that are published with the site
        full = write_manifest(sc)
        keep = {str(f.relative_to(sc.run_dir)).replace("\\", "/") for f in published}
        files = [x for x in full["files"] if x["path"] in keep]
        man = {"scenario": name, "files": files, "total_gb": sum(x["bytes"] for x in files) / 1e9,
               "root": f"the web copy (the complete {full['total_gb']:.1f} GB archive is listed while the workstation is online)"}
        (snap / "run" / name).mkdir(parents=True)
        (snap / "run" / name / "status.json").write_text(json.dumps(st))
        (snap / "run" / name / "manifest.json").write_text(json.dumps(man))

        meta = sc.load_meta() if sc.meta_file.exists() else {}
        listing.append({"name": name, "title": sc.cfg.get("title", name.replace("_", " ")), "description": sc["description"],
                        "reach_km": sc["domain"]["reach_length_km"],
                        "dam": {k: meta.get("dam", {}).get(k) for k in ("height_m", "crest_m")},
                        "lat": sc["dam"]["lat"], "lon": sc["dam"]["lon"], "type": sc["dam"].get("type", "dam"),
                        "reservoir_Mm3": round(meta.get("reservoir", {}).get("volume_m3", 0) / 1e6, 2),
                        "bbox": jalpravah._bbox(sc, meta),
                        "extents": {m: f"/run/{name}/files/results/{m}/gis/{m}_flood_extent.geojson" for m in done
                                    if (sc.results_dir / m / "gis" / f"{m}_flood_extent.geojson").exists()},
                        "status": st})
        print(f"[web] {name}: models {done}, {len(published)} files")
    snap.mkdir(exist_ok=True)
    (snap / "scenarios.json").write_text(json.dumps(listing))

    # the app pages, unchanged apart from the workstation bar
    for src, dst in (("home.html", "index.html"), ("builder.html", "builder.html"), ("satellite.html", "satellite.html")):
        html = (ASSETS / src).read_text(encoding="utf-8")
        html = replace_once(html, '<script src="/hydro.js"></script>\n', '<script src="/hydro.js"></script>\n' + REMOTE_TAG, src)
        (out / dst).write_text(html, encoding="utf-8")

    proxy = lambda p: {"source": f"{p}/:path*", "destination": f"/api/proxy?__p={p}/:path*"}  # noqa: E731
    vercel = {
        "functions": {"api/proxy.js": {"maxDuration": 60}},
        "rewrites": [
            {"source": "/favicon.ico", "destination": "/logo.png"},
            {"source": "/builder", "destination": "/builder.html"},
            {"source": "/satellite", "destination": "/satellite.html"},
            {"source": "/run/([^/]+)/", "destination": "/api/proxy?__p=/run/$1/"},  # scenario overview (trailing slash)
            proxy("/api"), proxy("/run"), proxy("/satellite-files"),
        ],
        "headers": [{"source": "/run/(.*)/files/(.*)\\.(tif|zip|kml|json|shp|shx|dbf|prj|cpg)",
                     "headers": [{"key": "Content-Disposition", "value": "attachment"}]}],
    }
    (out / "vercel.json").write_text(json.dumps(vercel, indent=2))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[web] wrote {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "web"))
    export(Path(ap.parse_args().out))
