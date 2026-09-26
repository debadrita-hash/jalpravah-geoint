"""
Model-independent result store, flood metrics and GIS export.

Every solver writes the same standard product in runs/<scenario>/results/<model>/:
    frames.npz   times [nt], depth [nt, ncell], u, v [nt, ncell]  (active analysis cells only)
    series.json  station discharge / depth, reservoir level, domain volume, outflow, run info
Metrics and exports are then computed identically for SPH and Delft3D.
"""
import json
import math
from pathlib import Path

import numpy as np

from .terrain import LocalFrame, load_terrain

HAZARD_CLASSES = [(0.0, "Low"), (0.75, "Moderate - danger for some"), (1.25, "Significant - danger for most"),
                  (2.0, "Extreme - danger for all")]


class ResultStore:
    def __init__(self, sc, model):
        self.sc, self.model = sc, model
        self.dir = sc.results_dir / model
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t = load_terrain(sc)
        self.meta = sc.load_meta()
        self.active = self.t["cell_domain"]
        self.idx = np.nonzero(self.active.ravel())[0]

    # -- writing
    def save_frames(self, times, depth, u, v):
        np.savez_compressed(self.dir / "frames.npz", times=np.asarray(times, np.float32),
                            depth=np.asarray(depth, np.float32), u=np.asarray(u, np.float16), v=np.asarray(v, np.float16))

    def save_series(self, series):
        with open(self.dir / "series.json", "w", encoding="utf-8") as f:
            json.dump(series, f, indent=1)

    # -- reading
    def load(self):
        fr = np.load(self.dir / "frames.npz")
        with open(self.dir / "series.json", encoding="utf-8") as f:
            series = json.load(f)
        return {k: fr[k] for k in fr.files}, series

    def to_grid(self, vals, fill=np.nan):
        g = np.full(self.active.size, fill, np.float32)
        g[self.idx] = vals
        return g.reshape(self.active.shape)


# ----------------------------------------------------------------------------- metrics
def compute_metrics(store, t_max=None, suffix=""):
    """Event metrics over [0, t_max] (whole run by default); writes metrics{suffix}.json / max_fields{suffix}.npz."""
    sc, meta = store.sc, store.meta
    sim = sc["simulation"]
    fr, series = store.load()
    T = fr["times"].astype(float)
    keep = T <= (t_max + 1e-6 if t_max is not None else np.inf)
    T = T[keep]
    h = fr["depth"][keep].astype(np.float32)
    spd = np.hypot(fr["u"][keep].astype(np.float32), fr["v"][keep].astype(np.float32))
    nk = len(T)
    for st_ in series["stations"]:
        st_["time"], st_["discharge"] = st_["time"][:nk], st_["discharge"][:nk]
    mb0 = series["mass_balance"]
    out_t = np.asarray(series["outflow"]["time"], float)
    out_c = np.asarray(series["outflow"]["cumulative_m3"], float)
    series["mass_balance"] = dict(mb0, final_volume_m3=float(h[-1].sum() * meta["analysis_grid"]["dx"] ** 2)
                                  if t_max is not None else mb0["final_volume_m3"],
                                  outflow_volume_m3=float(np.interp(T[-1], out_t, out_c)))
    if t_max is not None and "lost_cumulative_m3" in mb0:
        series["mass_balance"]["lost_volume_m3"] = float(np.interp(T[-1], out_t, mb0["lost_cumulative_m3"]))
        series["mass_balance"]["final_volume_m3"] = float(np.interp(T[-1], out_t, mb0["volume_in_domain_m3"]))
    wet_thr, arr_thr = sim["wet_threshold_m"], sim["arrival_threshold_m"]
    dxa = meta["analysis_grid"]["dx"]
    area = dxa * dxa
    resv = store.t["cell_reservoir"].ravel()[store.idx]
    ch = _cell_chainage(store)
    downstream = ~resv & (ch > 0.5 * meta["dam"]["crest_width_m"] + meta["dam"]["height_m"] * meta["dam"]["downstream_slope"])  # past the dam toe

    # time maxima computed in slices of frames, all in float32 (keeps peak memory low)
    nT, nC = h.shape
    hmax = np.zeros(nC, np.float32); vmax = np.zeros(nC, np.float32); hv = np.zeros(nC, np.float32); hr = np.zeros(nC, np.float32)
    first = np.full(nC, -1, np.int64); t_hmax_i = np.zeros(nC, np.int64); duration = np.zeros(nC, np.float32)
    dt = np.diff(T, prepend=T[0]).astype(np.float32)
    for a in range(0, nT, 40):
        hs = h[a:a + 40]
        wet_s = hs >= wet_thr
        sp = np.where(wet_s, spd[a:a + 40], np.float32(0))
        vmax = np.maximum(vmax, sp.max(0))
        hv = np.maximum(hv, (hs * sp).max(0))
        rating = np.where(wet_s, hs * (sp + np.float32(0.5)) + np.where(hs > 0.25, np.float32(0.5), np.float32(0)), np.float32(0))
        hr = np.maximum(hr, rating.max(0))  # DEFRA FD2320 hazard rating
        k = hs.argmax(0)
        better = hs[k, np.arange(nC)] > hmax
        t_hmax_i = np.where(better, a + k, t_hmax_i)
        hmax = np.maximum(hmax, hs.max(0))
        arr_s = hs >= arr_thr
        hit = arr_s.any(0) & (first < 0)
        first = np.where(hit, a + arr_s.argmax(0), first)
        duration += (wet_s * dt[a:a + 40, None]).sum(0)
        del hs, wet_s, sp, rating, arr_s
    reached = first >= 0
    arrival = np.where(reached, T[np.maximum(first, 0)], np.nan)
    t_hmax = T[t_hmax_i]
    flooded = (hmax >= wet_thr) & ~resv

    area_t = np.array([((fr_ >= wet_thr) & ~resv).sum() for fr_ in h]) * area
    vol_t = h.sum(1, dtype=np.float64) * area
    front = np.array([ch[(fr_ >= arr_thr) & downstream].max() if ((fr_ >= arr_thr) & downstream).any() else 0.0 for fr_ in h])

    haz_cls = np.digitize(hr, [c for c, _ in HAZARD_CLASSES][1:])
    haz_area = {name: float(((haz_cls == k) & flooded).sum() * area / 1e6) for k, (_, name) in enumerate(HAZARD_CLASSES)}

    lookup = np.full(store.active.shape, -1, np.int64)
    lookup.ravel()[store.idx] = np.arange(len(store.idx))
    st, st_depth = [], []
    for k, s in enumerate(meta["stations"]):
        a, b = s["cell_range"]
        if s["axis"] == "x":
            cells = lookup[a:b + 1, s["face_index"] - 1]
        else:
            cells = lookup[s["face_index"] - 1, a:b + 1]
        cells = cells[cells >= 0]
        hs = h[:, cells].max(1) if len(cells) else np.zeros(len(T))
        st_depth.append(hs)
        q = np.asarray(series["stations"][k]["discharge"], float)
        tq = np.asarray(series["stations"][k]["time"], float)
        ip = int(np.argmax(q)) if len(q) else 0
        arr = T[np.argmax(hs >= arr_thr)] if (hs >= arr_thr).any() else None
        st.append({"name": s["name"], "chainage_km": s["chainage_m"] / 1000.0,
                   "peak_discharge_m3s": float(q[ip]) if len(q) else 0.0,
                   "time_of_peak_discharge_min": float(tq[ip] / 60.0) if len(q) else None,
                   "peak_depth_m": float(hs.max()),
                   "time_of_peak_depth_min": float(T[int(np.argmax(hs))] / 60.0),
                   "arrival_time_min": None if arr is None else float(arr / 60.0),
                   "passed_volume_Mm3": float(np.trapz(np.clip(q, 0, None), tq) / 1e6) if len(q) > 1 else 0.0})

    tv = np.array(meta["dam"]["axis_tangent"])
    gx, gy = -300.0 * tv
    ag = meta["analysis_grid"]
    gi, gj = int((gx - ag["x0_corner"]) // dxa), int((gy - ag["y0_corner"]) // dxa)
    gcell = lookup[gj, gi]
    rl = store.t["cell_z"][gj, gi] + h[:, gcell]
    rt = T
    crest, bed = meta["dam"]["crest_m"], meta["dam"]["bed_m"]
    half = crest - 0.5 * (crest - bed)
    t_half = float(rt[np.argmax(rl <= half)] / 60.0) if (rl <= half).any() else None

    v0 = float(series["mass_balance"]["initial_volume_m3"])
    mb = series["mass_balance"]
    metrics = {
        "model": store.model,
        "flooded_area_km2": float(flooded.sum() * area / 1e6),
        "max_depth_m": float(hmax[downstream].max()),
        "max_velocity_ms": float(vmax[downstream].max()),
        "max_unit_discharge_m2s": float(hv[downstream].max()),
        "max_front_distance_km": float(front.max() / 1000.0),
        "front_arrival_min": {f"{km:g} km": _first_time(T, front >= km * 1000) for km in (1, 5, 10, 15, 20, 25)},
        "hazard_area_km2": haz_area,
        "breach_peak_discharge_m3s": st[0]["peak_discharge_m3s"],
        "breach_time_to_peak_min": st[0]["time_of_peak_discharge_min"],
        "reservoir_half_drawdown_min": t_half,
        "reservoir_final_level_m": float(rl[-1]) if len(rl) else None,
        "stations": st,
        "mass_balance": {"initial_volume_m3": v0, "final_volume_in_domain_m3": float(mb["final_volume_m3"]),
                         "outflow_volume_m3": float(mb["outflow_volume_m3"]),
                         "lost_volume_m3": float(mb.get("lost_volume_m3", 0.0)),
                         "error_pct": float(100.0 * (v0 - mb["final_volume_m3"] - mb["outflow_volume_m3"]) / v0)},
        "run": series["run_info"],
    }
    fields = {"hmax": hmax, "vmax": vmax, "hv": hv, "hazard_rating": hr, "hazard_class": haz_cls.astype(np.float32),
              "arrival_min": arrival / 60.0, "time_of_hmax_min": t_hmax / 60.0, "flood_duration_min": duration / 60.0}
    np.savez_compressed(store.dir / f"max_fields{suffix}.npz", **fields)
    ts = {"time_min": (T / 60.0).tolist(), "flooded_area_km2": (area_t / 1e6).tolist(),
          "reservoir_level_m": rl.tolist(), "station_depth_m": [d.tolist() for d in st_depth],
          "station_discharge_m3s": [series["stations"][k]["discharge"] for k in range(len(st))],
          "volume_in_domain_Mm3": (vol_t / 1e6).tolist(), "front_km": (front / 1000.0).tolist()}
    with open(store.dir / f"metrics{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(metrics | {"timeseries": ts}, f, indent=1)
    return metrics, fields


def _first_time(T, cond):
    return float(T[np.argmax(cond)] / 60.0) if cond.any() else None


def _cell_chainage(store):
    from scipy.spatial import cKDTree
    m = store.meta["analysis_grid"]
    th = store.t["thalweg"]
    xc, yc = np.meshgrid(store.t["xc"], store.t["yc"])
    pts = np.c_[xc.ravel()[store.idx], yc.ravel()[store.idx]]
    _, k = cKDTree(th[:, :2]).query(pts)
    return th[k, 2]


# ----------------------------------------------------------------------------- GIS export
def export_gis(store, fields, out_dir=None):
    """North-up UTM GeoTIFFs + shapefile / KML / GeoJSON polygons of the flood extent and hazard classes."""
    import rasterio
    import shapefile
    import simplekml
    from pyproj import CRS, Transformer
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from shapely.geometry import mapping, shape
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union

    out = Path(out_dir or store.dir / "gis")
    out.mkdir(parents=True, exist_ok=True)
    meta = store.meta
    fr = LocalFrame.from_dict(meta["frame"])
    ag = meta["analysis_grid"]
    dx = ag["dx"]
    # UTM raster covering the analysis grid
    cx = [ag["x0_corner"], ag["x0_corner"] + ag["nx"] * dx]
    cy = [ag["y0_corner"], ag["y0_corner"] + ag["ny"] * dx]
    X, Y = fr.to_utm(np.array([cx[0], cx[1], cx[0], cx[1]]), np.array([cy[0], cy[0], cy[1], cy[1]]))
    xmin, xmax, ymin, ymax = X.min(), X.max(), Y.min(), Y.max()
    nx, ny = int(math.ceil((xmax - xmin) / dx)), int(math.ceil((ymax - ymin) / dx))
    tr = from_origin(xmin, ymax, dx, dx)
    UX, UY = np.meshgrid(xmin + (np.arange(nx) + 0.5) * dx, ymax - (np.arange(ny) + 0.5) * dx)
    lx, ly = fr.to_local(UX, UY)
    ii = np.floor((lx - ag["x0_corner"]) / dx).astype(int)
    jj = np.floor((ly - ag["y0_corner"]) / dx).astype(int)
    inside = (ii >= 0) & (ii < ag["nx"]) & (jj >= 0) & (jj < ag["ny"])

    def to_utm_raster(vals):
        g = store.to_grid(vals)
        r = np.full((ny, nx), np.nan, np.float32)
        r[inside] = g[jj[inside], ii[inside]]
        return r

    crs = meta["crs"]
    thr = store.sc["simulation"]["wet_threshold_m"]
    resv = store.t["cell_reservoir"].ravel()[store.idx]
    rasters = {}
    for name, vals in fields.items():
        vals = np.where((fields["hmax"] >= thr) & ~resv, vals, np.nan)
        r = to_utm_raster(vals)
        rasters[name] = r
        with rasterio.open(out / f"{store.model}_{name}.tif", "w", driver="GTiff", width=nx, height=ny, count=1,
                           dtype="float32", crs=crs, transform=tr, nodata=np.nan, compress="deflate") as ds:
            ds.write(r, 1)

    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    wkt_wgs = CRS.from_epsg(4326).to_wkt(version="WKT1_ESRI")

    def polygons(mask):
        geoms = [shape(g) for g, val in shapes(mask.astype(np.uint8), mask=mask, transform=tr) if val == 1]
        return unary_union(geoms) if geoms else None

    def reproject_geom(g):
        from shapely.ops import transform as stransform
        return stransform(lambda x, y, z=None: to_wgs.transform(x, y), g)

    layers = {"flood_extent": [(polygons(np.isfinite(rasters["hmax"])), {"class": "Inundated", "min_depth_m": thr})]}
    hz = rasters["hazard_class"]
    layers["hazard"] = [(polygons(hz == k), {"class": name, "min_rating": lo}) for k, (lo, name) in enumerate(HAZARD_CLASSES)]
    geojson_all = {}
    for lname, items in layers.items():
        items = [(reproject_geom(g.simplify(dx * 0.5)), p) for g, p in items if g is not None and not g.is_empty]
        base = out / f"{store.model}_{lname}"
        w = shapefile.Writer(str(base), shapeType=shapefile.POLYGON)
        w.field("model", "C", 20)
        w.field("class", "C", 40)
        w.field("area_km2", "N", 12, 4)
        feats = []
        kml = simplekml.Kml(name=f"{store.model} {lname}")
        colors = ["7f00ff00", "7f00ffff", "7f0080ff", "7f0000ff"]
        for k, (g, p) in enumerate(items):
            area_km2 = _area_km2(g, crs, to_wgs)
            polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
            parts = []
            for pg in polys:
                pg = orient(pg, sign=-1.0)  # shapefile: outer rings clockwise, holes counter-clockwise
                parts.append([list(c) for c in pg.exterior.coords])
                parts += [[list(c) for c in r.coords] for r in pg.interiors]
                pk = kml.newpolygon(name=p["class"], outerboundaryis=list(pg.exterior.coords),
                                    innerboundaryis=[list(r.coords) for r in pg.interiors])
                pk.style.polystyle.color = colors[k % 4] if lname == "hazard" else "7fff7f00"
                pk.style.linestyle.width = 0.5
            w.poly(parts)
            w.record(store.model, p["class"], round(area_km2, 4))
            feats.append({"type": "Feature", "geometry": mapping(g), "properties": {**p, "model": store.model, "area_km2": area_km2}})
        w.close()
        (base.with_suffix(".prj")).write_text(wkt_wgs)
        kml.save(str(base.with_suffix(".kml")))
        gj = {"type": "FeatureCollection", "features": feats}
        (base.with_suffix(".geojson")).write_text(json.dumps(gj))
        geojson_all[lname] = gj
    return out, geojson_all


def _area_km2(g_wgs, crs, to_wgs):
    from pyproj import Transformer
    from shapely.ops import transform as stransform
    back = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return float(stransform(lambda x, y, z=None: back.transform(x, y), g_wgs).area / 1e6)
