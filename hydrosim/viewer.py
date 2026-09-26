"""
Builds the result viewer: runs/<scenario>/viewer/{sph.html, delft3d.html, comparison.html} plus data/*.js.

Data ships as JavaScript files (gzip + base64) so the pages open straight from disk (file://) and can also
be published as-is. Frames are sampled every `view_interval_s` and quantised to 8 bits:
depth 0.25 m steps (0 = dry below the wet threshold), speed 0.15 m/s steps.
"""
import base64
import gzip
import io
import json
import math
import shutil
from pathlib import Path

import numpy as np

from .results import HAZARD_CLASSES, ResultStore
from .terrain import LocalFrame

ASSETS = Path(__file__).resolve().parent / "viewer_assets"
DEPTH_STEP, SPEED_STEP = 0.25, 0.15


def _b64(arr):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(arr).tobytes(), 6)).decode()


def _js(path, name, obj):
    path.write_text(f"window.HYDRO_DATA=window.HYDRO_DATA||{{}};window.HYDRO_DATA[{json.dumps(name)}]={json.dumps(obj, separators=(',', ':'))};\n",
                    encoding="utf-8")


def _terrain_png(t, meta, sc=None):
    """Base map shared by the 2-D and 3-D views: Sentinel-2 natural colour (when available) blended with a
    10 m hillshade of the DEM; dam embankment in rust, breach notch in amber. JPEG, analysis-grid extent."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import LightSource
    from PIL import Image
    ag = meta["analysis_grid"]
    f = int(round(ag["dx"] / meta["resolution_m"]))
    z = t["z"][: ag["ny"] * f, : ag["nx"] * f]
    ny, nx = z.shape
    zc = np.where(z < 8000, z, np.nan)
    zc = np.where(np.isfinite(zc), zc, np.nanmedian(zc))
    shade = LightSource(azdeg=315, altdeg=35).hillshade(zc, vert_exag=1.5, dx=10, dy=10)
    rgb = None
    if sc is not None:
        try:
            import rasterio
            from rasterio.warp import transform as wt
            from scipy import ndimage
            if "satellite" not in sc["inputs"]:
                raise KeyError("no imagery configured")
            with rasterio.open(sc.input_path("satellite")) as ds:
                rgb8 = ds.count == 3 and ds.dtypes[0] == "uint8"  # Sentinel-2 true-colour (TCI) product
                bands = ds.read([1, 2, 3] if rgb8 else [3, 2, 1]).astype(np.float32)
                inv = ~ds.transform
                fr = LocalFrame.from_dict(meta["frame"])
                chans = [np.zeros((ny, nx), np.float32) for _ in range(3)]
                for r0 in range(0, ny, 200):  # row blocks keep memory small
                    r1 = min(ny, r0 + 200)
                    X, Y = np.meshgrid(t["xs"][:nx], t["ys"][r0:r1])
                    U, V = fr.to_utm(X, Y)
                    lon, lat = wt(meta["crs"], ds.crs, U.ravel(), V.ravel())
                    col, row = inv * (np.asarray(lon), np.asarray(lat))
                    for c in range(3):
                        chans[c][r0:r1] = ndimage.map_coordinates(bands[c], [np.asarray(row) - 0.5, np.asarray(col) - 0.5],
                                                                  order=1, mode="nearest").reshape(r1 - r0, nx)
            img = np.stack(chans, -1)
            img = np.clip(img / 255.0, 0, 1) if rgb8 else np.clip(img / 1800.0, 0, 1) ** (1 / 1.6)  # reflectance stretch
            rgb = img
        except Exception as ex:
            print("[viewer] satellite texture skipped:", ex)
    if rgb is None:
        from matplotlib.colors import LinearSegmentedColormap
        hyps = LinearSegmentedColormap.from_list("h", ["#9fb88a", "#c9c79a", "#b8a27f", "#9b8b78", "#d9d5cf"])
        zr = (zc - np.percentile(zc, 1)) / (np.percentile(zc, 99.5) - np.percentile(zc, 1))
        rgb = hyps(np.clip(zr, 0, 1))[..., :3]
    out = np.clip(rgb * (0.45 + 0.75 * shade[..., None]), 0, 1)
    dam = t["dam"][:ny, :nx] & ~t["notch"][:ny, :nx]
    notch = t["notch"][:ny, :nx] & t["dam"][:ny, :nx]
    out[dam] = [0.55, 0.20, 0.13]
    out[notch] = [0.96, 0.72, 0.16]
    img = Image.fromarray((out[::-1] * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=86)
    return base64.b64encode(buf.getvalue()).decode()


def _scenario_data(sc):
    from .exposure import assets_local
    store = ResultStore(sc, "delft3d")
    t, meta = store.t, store.meta
    ag = meta["analysis_grid"]
    grid = {"nx": ag["nx"], "ny": ag["ny"], "dx": ag["dx"], "x0": ag["x0_corner"], "y0": ag["y0_corner"]}
    th = t["thalweg"]
    th = th[(th[:, 2] > -3000)][::3]
    dom = t["domain"]
    xs, ys = t["xs"], t["ys"]

    def in_domain(x, y):
        i = np.clip(((x - xs[0]) / meta["resolution_m"]).astype(int), 0, len(xs) - 1)
        j = np.clip(((y - ys[0]) / meta["resolution_m"]).astype(int), 0, len(ys) - 1)
        return dom[j, i]

    roads, buildings, places = [], [], []
    try:
        # map overlay uses OSM data only once the analysis has downloaded it (never blocks a build)
        osm_assets = assets_local(sc) if (sc.scenario_dir / "osm_exposure.json").exists() else []
        for a in osm_assets:
            p = a["pts"]
            inside = in_domain(p[:, 0], p[:, 1])
            if not inside.any():
                continue
            if a["kind"] in ("road", "bridge"):
                roads.append(np.round(p[::3], 0).tolist())
            elif a["kind"] in ("building", "facility"):
                buildings.append([round(float(p[0, 0]), 0), round(float(p[0, 1]), 0)])
            elif a["kind"] == "settlement" and a["name"]:
                places.append([round(float(p[0, 0]), 0), round(float(p[0, 1]), 0), a["name"]])
    except Exception as ex:  # exposure layer is optional for rendering
        print("[viewer] OSM layer skipped:", ex)
    return {
        "name": sc.name, "title": sc.cfg.get("title", sc.name.replace("_", " ")), "description": sc["description"], "grid": grid,
        "cells_b64": _b64(store.idx.astype(np.int32)),
        "cell_z_b64": _b64(t["cell_z"].ravel()[store.idx].astype(np.float32)),
        "reservoir_b64": _b64(t["cell_reservoir"].ravel()[store.idx].astype(np.uint8)),
        "terrain_png": _terrain_png(t, meta, sc),
        "thalweg": np.round(th[:, :4], 1).tolist(),
        "stations": [{k: s[k] for k in ("name", "chainage_m", "x", "y", "section")} for s in meta["stations"]],
        "dam": meta["dam"], "breach": meta["breach"], "reservoir": meta["reservoir"],
        "frame": meta["frame"], "crs": meta["crs"],
        "roads": roads, "buildings": buildings, "places": places,
        "hazard_classes": [n for _, n in HAZARD_CLASSES],
        "thresholds": {"wet": sc["simulation"]["wet_threshold_m"], "arrival": sc["simulation"]["arrival_threshold_m"]},
        "depth_step": DEPTH_STEP, "speed_step": SPEED_STEP,
    }


def _model_data(sc, model, view_interval_s=60.0):
    store = ResultStore(sc, model)
    fr = np.load(store.dir / "frames.npz")
    T = fr["times"].astype(float)
    step = max(1, int(round(view_interval_s / max(T[1] - T[0], 1e-9)))) if len(T) > 1 else 1
    sel = np.arange(0, len(T), step)
    h = fr["depth"][sel].astype(np.float32)
    spd = np.hypot(fr["u"][sel].astype(np.float32), fr["v"][sel].astype(np.float32))
    thr = sc["simulation"]["wet_threshold_m"]
    wet = h >= thr
    hq = np.where(wet, np.clip(np.maximum(1, np.round(h / DEPTH_STEP)), 1, 255), 0).astype(np.uint8)
    vq = np.where(wet, np.clip(np.round(spd / SPEED_STEP), 0, 255), 0).astype(np.uint8)
    mf = np.load(store.dir / "max_fields.npz")
    metrics = json.loads((store.dir / "metrics.json").read_text(encoding="utf-8"))
    imp = store.dir / "impacts.json"
    return {
        "model": model, "times": T[sel].tolist(), "nframes": int(len(sel)),
        "depth_b64": _b64(hq), "speed_b64": _b64(vq),
        "max": {k: _b64(mf[k].astype(np.float32)) for k in ("hmax", "vmax", "hazard_class", "arrival_min", "hv")},
        "metrics": metrics,
        "impacts": json.loads(imp.read_text(encoding="utf-8")) if imp.exists() else None,
        "gis_dir": str(store.dir / "gis"),
        "planned_s": float(sc["sph"]["duration_s"] if model == "sph" else sc["simulation"]["duration_s"]),
    }


def _terrain3d(sc, out):
    """40 m height grid over the analysis extent for the 3-D views (gzip float32)."""
    t, meta = ResultStore(sc, "delft3d").t, sc.load_meta()
    ag, res = meta["analysis_grid"], meta["resolution_m"]
    f, r = 4, int(round(ag["dx"] / res))
    z = t["z"][: ag["ny"] * r, : ag["nx"] * r]
    zz = z[::f, ::f].astype(np.float32)
    valid = zz < 8000  # DEM no-data outside the tile is stored as 9999
    zz = np.ascontiguousarray(np.where(valid, zz, np.median(zz[valid])).astype(np.float32))
    (out / "data" / "terrain3d.bin").write_bytes(gzip.compress(zz.tobytes(), 6))
    return {"file": "terrain3d.bin", "nx": int(zz.shape[1]), "ny": int(zz.shape[0]), "dx": res * f,
            "x0": float(t["xs"][0]), "y0": float(t["ys"][0]), "zmin": float(np.percentile(zz, 0.5)) - 20.0}


def _dam_patch(sc, out):
    t, meta = ResultStore(sc, "delft3d").t, sc.load_meta()
    xs, ys = t["xs"], t["ys"]
    i0, i1 = np.searchsorted(xs, -700), np.searchsorted(xs, 700)
    j0, j1 = np.searchsorted(ys, -700), np.searchsorted(ys, 700)
    zz = np.ascontiguousarray(t["z"][j0:j1, i0:i1].astype(np.float32))
    (out / "data" / "dam_patch.bin").write_bytes(gzip.compress(zz.tobytes(), 6))
    return {"file": "dam_patch.bin", "nx": int(zz.shape[1]), "ny": int(zz.shape[0]), "dx": float(meta["resolution_m"]),
            "x0": float(xs[i0]), "y0": float(ys[j0])}


def _particles3d(sc, out):
    p = ResultStore(sc, "sph").dir / "particles3d.npz"
    if not p.exists():
        return None
    d = np.load(p)
    (out / "data" / "sph_particles_xyz.bin").write_bytes(gzip.compress(d["xyz"].astype(np.uint16).tobytes(), 6))
    (out / "data" / "sph_particles_speed.bin").write_bytes(gzip.compress(d["speed"].astype(np.uint8).tobytes(), 6))
    return {"file_xyz": "sph_particles_xyz.bin", "file_speed": "sph_particles_speed.bin", "times": d["times"].tolist(),
            "counts": d["counts"].tolist(), "origin": d["origin"].tolist(), "scale": d["scale"].tolist()}


def build(sc, models=("sph", "delft3d"), view_interval_s=60.0):
    out = sc.viewer_dir
    (out / "data").mkdir(parents=True, exist_ok=True)
    for f in ("hydro.css", "hydro.js"):
        shutil.copy(ASSETS / f, out / f)
    scen = _scenario_data(sc)
    scen["terrain3d"] = _terrain3d(sc, out)
    scen["dam_patch"] = _dam_patch(sc, out)
    _js(out / "data" / "scenario.js", "scenario", scen)
    for m in models:
        if (ResultStore(sc, m).dir / "metrics.json").exists():
            md = _model_data(sc, m, view_interval_s)
            if m == "sph":
                md["particles3d"] = _particles3d(sc, out)
            _js(out / "data" / f"{m}.js", m, md)
    cmp = sc.results_dir / "comparison.json"
    if cmp.exists():
        cf = np.load(sc.results_dir / "comparison_fields.npz")
        _js(out / "data" / "comparison.js", "comparison",
            json.loads(cmp.read_text(encoding="utf-8")) | {
                "agreement_b64": _b64(cf["agreement"].astype(np.uint8)),
                "hdiff_b64": _b64(np.nan_to_num(cf["hmax_difference"], nan=-999).astype(np.float32))})
    for page in ("model.html", "comparison.html", "index.html", "view3d.html"):
        src = "<!doctype html>\n<meta charset=\"utf-8\">\n" + (ASSETS / page).read_text(encoding="utf-8")
        if page == "view3d.html":
            for m, label in (("sph", "SPH"), ("delft3d", "Delft3D")):
                (out / f"{m}3d.html").write_text(src.replace("__MODEL__", m).replace("__LABEL__", label), encoding="utf-8")
            continue
        if page == "model.html":
            for m, label in (("sph", "SPH"), ("delft3d", "Delft3D")):
                (out / f"{m}.html").write_text(src.replace("__MODEL__", m).replace("__LABEL__", label), encoding="utf-8")
        else:
            (out / page).write_text(src, encoding="utf-8")
    return out
