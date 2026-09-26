"""
Static, read-only copy of JalPravah for web hosting (Vercel): the finished scenarios with their 2-D and 3-D
views, comparison, damage estimate and GIS downloads.

    python export_web.py [--out web]

Nothing here runs a solver. Building scenarios, running Delft3D / SPH and Earth Engine analyses need Docker
and the local server, so the hosted copy says so instead of offering those buttons. Only scenarios with at
least one finished model are exported, and only files that exist are linked.
"""
import argparse
import json
import shutil
from pathlib import Path

import jalpravah
from hydrosim.archive import write_manifest

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "hydrosim" / "viewer_assets"

# result files published per model (the multi-GB solver cases and raw output stay on the workstation)
MODEL_FILES = ["metrics.json", "impacts.json", "series.json"]
GIS_EXT = {".tif", ".geojson", ".kml", ".shp", ".shx", ".dbf", ".prj", ".cpg"}
SCENARIO_GIS_EXT = {".geojson", ".kml", ".shp", ".shx", ".dbf", ".prj", ".cpg"}

BANNER = """<div class="wrap" style="padding-bottom:0"><p class="note" style="background:var(--surface-2);border:1px solid var(--rule);border-radius:6px;padding:10px 14px;max-width:none">
Hosted results viewer. These are the finished runs of both solvers; new scenarios, simulations and satellite analyses run on the JalPravah workstation
(<b>python jalpravah.py</b>, Docker with DualSPHysics and Delft3D-FLOW).</p></div>"""
HIDE_CSS = "<style>[data-run],[data-del],.confirm,section.panel:has(#jobs){display:none!important}</style>"


def replace_once(text, old, new, where):
    if old not in text:
        raise SystemExit(f"export_web: expected text not found in {where}: {old[:60]!r}")
    return text.replace(old, new, 1)


def notice_page(src, here, eyebrow, title, body):
    """The page's own head and app bar, then an explanation of what it needs to run."""
    html = (ASSETS / src).read_text(encoding="utf-8")
    head = html[:html.index('<div class="wrap">')]
    return head + f"""<div class="wrap">
  <div class="pagehead"><div><span class="eyebrow">{eyebrow}</span><h1>{title}</h1>{body}</div></div>
  <p><a class="btn primary" href="/" style="font:600 13px var(--font-head);padding:8px 14px;border-radius:4px;background:var(--d3d);color:#fff;text-decoration:none;display:inline-block">Open the finished scenarios</a></p>
</div>
"""


def export(out):
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():  # keep .vercel, .env.local (the linked Vercel project)
        if not old.name.startswith("."):
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    for f in ("hydro.css", "hydro.js", "logo.png"):
        shutil.copy2(ASSETS / f, out / f)

    status = jalpravah.Status()
    listing = []
    for name, sc in jalpravah.scenarios().items():
        st = status.get(sc)
        done = [m for m in ("delft3d", "sph") if st[m]["state"] == "complete"]
        if not done:
            print(f"[web] skipping {name}: no finished model run")
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

        full = write_manifest(sc)
        keep = {str(f.relative_to(sc.run_dir)).replace("\\", "/") for f in published}
        files = [x for x in full["files"] if x["path"] in keep]
        man = {"scenario": name, "files": files, "total_gb": sum(x["bytes"] for x in files) / 1e9,
               "root": f"this hosted copy (the complete {full['total_gb']:.1f} GB archive with solver cases and raw output is kept on the workstation)"}
        (run / "api").mkdir()
        (run / "api" / "status.json").write_text(json.dumps(st))
        (run / "api" / "manifest.json").write_text(json.dumps(man))

        # model pages: link only the published files; overview: size in MB
        for page in ("sph.html", "delft3d.html"):
            p = run / page
            t = p.read_text(encoding="utf-8")
            t = replace_once(t, '["All depth/velocity frames (NumPy)", "frames.npz"], ', "", page)
            p.write_text(t, encoding="utf-8")
        p = run / "index.html"
        t = replace_once(p.read_text(encoding="utf-8"), "${fmt(man.total_gb, 1)} GB", "${fmt(man.total_gb * 1000, 1)} MB", "index.html")
        p.write_text(t, encoding="utf-8")

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

    (out / "api").mkdir()
    (out / "api" / "scenarios.json").write_text(json.dumps(listing))
    (out / "api" / "jobs.json").write_text("[]")

    home = (ASSETS / "home.html").read_text(encoding="utf-8")
    home = replace_once(home, "</style>", "</style>\n" + HIDE_CSS, "home.html")
    home = replace_once(home, "</header>\n", "</header>\n" + BANNER + "\n", "home.html")
    (out / "index.html").write_text(home, encoding="utf-8")

    (out / "builder.html").write_text(notice_page(
        "builder.html", "builder", "Scenario builder · open DEM, imagery and OpenStreetMap", "New scenario",
        """<p class="lede">Building a scenario downloads the Copernicus GLO-30 DEM and Sentinel-2 imagery for the chosen dam,
derives the reservoir and Froehlich breach, and prepares both solver cases. Running it takes hours of CPU in Docker
(DualSPHysics and Delft3D-FLOW), so it is done on the JalPravah workstation, not on this web host.</p>
<p class="lede">On the workstation: <b>python jalpravah.py</b>, then open <b>New scenario</b> at http://localhost:8765/.</p>"""), encoding="utf-8")
    (out / "satellite.html").write_text(notice_page(
        "satellite.html", "satellite", "Near-real-time analysis · Google Earth Engine · open Sentinel-1 radar", "Satellite flood mapping",
        """<p class="lede">Flood mapping compares Sentinel-1 VV radar before and after an event on Google Earth Engine
(ΔVV ≤ −3 dB, VV &lt; −15 dB, slope &lt; 5°, permanent water from JRC masked). It signs in with your own
Google account and Earth Engine Cloud project, which is kept on the JalPravah workstation, so it runs there.</p>
<p class="lede">On the workstation: <b>python jalpravah.py</b>, then open <b>Satellite flood mapping</b>.</p>"""), encoding="utf-8")

    vercel = {
        "cleanUrls": True,
        "rewrites": [
            {"source": "/api/scenarios", "destination": "/api/scenarios.json"},
            {"source": "/api/jobs", "destination": "/api/jobs.json"},
            {"source": "/favicon.ico", "destination": "/logo.png"},
            {"source": "/run/:s/api/:f", "destination": "/run/:s/api/:f.json"},
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
