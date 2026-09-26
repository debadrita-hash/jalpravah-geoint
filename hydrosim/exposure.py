"""
Exposure data for loss & damage: OpenStreetMap buildings, roads, bridges and settlements inside the
simulation corridor (Overpass API, cached in runs/<scenario>/scenario/osm_exposure.json), and the
intersection of those assets with each model's flood fields.
"""
import json
import math
import urllib.parse
import urllib.request

import numpy as np

from .terrain import LocalFrame

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter"]


def corridor_bbox_wgs84(meta):
    from pyproj import Transformer
    fr = LocalFrame.from_dict(meta["frame"])
    ag = meta["analysis_grid"]
    xs = [ag["x0_corner"], ag["x0_corner"] + ag["nx"] * ag["dx"]]
    ys = [ag["y0_corner"], ag["y0_corner"] + ag["ny"] * ag["dx"]]
    X, Y = fr.to_utm(np.array([xs[0], xs[1], xs[0], xs[1]]), np.array([ys[0], ys[0], ys[1], ys[1]]))
    lon, lat = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True).transform(X, Y)
    return float(min(lat)), float(min(lon)), float(max(lat)), float(max(lon))


def _overpass(q, log):
    import time as _time
    for attempt in range(4):
        for url in OVERPASS:
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                             headers={"User-Agent": "hydrosim-dambreak-research/1.0"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.loads(r.read().decode())
                if "remark" in d and ("error" in d["remark"].lower() or "timed out" in d["remark"].lower()):
                    log(f"[exposure] partial result from {url} ({d['remark'][:60]}); retrying")
                    continue
                return d
            except Exception as ex:
                log(f"[exposure] {url} failed: {ex}")
        _time.sleep(15 * (attempt + 1))
    raise RuntimeError("OpenStreetMap Overpass servers did not answer (they may be overloaded); retry later")


def fetch_osm(sc, log=print):
    """OSM exposure for the corridor, fetched in ~0.1 degree tiles (light queries) and cached."""
    cache = sc.scenario_dir / "osm_exposure.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    s, w, n, e = corridor_bbox_wgs84(sc.load_meta())
    step = 0.1
    seen, elements = set(), []
    lats = np.arange(s, n, step)
    lons = np.arange(w, e, step)
    t = 0
    for la in lats:
        for lo in lons:
            t += 1
            bb = f"{la:.5f},{lo:.5f},{min(la + step, n):.5f},{min(lo + step, e):.5f}"
            q = f"""[out:json][timeout:180];
(
  way["building"]({bb});
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|track|service)$"]({bb});
  way["bridge"="yes"]({bb});
  node["place"~"^(city|town|village|hamlet)$"]({bb});
  node["amenity"~"^(school|hospital|clinic|health_post)$"]({bb});
  way["amenity"~"^(school|hospital|clinic)$"]({bb});
  way["power"="plant"]({bb});
);
out geom;"""
            d = _overpass(q, log)
            for el in d["elements"]:
                key = (el["type"], el["id"])
                if key not in seen:
                    seen.add(key)
                    elements.append(el)
            log(f"[exposure] tile {t}/{len(lats) * len(lons)}: {len(elements)} features so far")
    data = {"elements": elements, "bbox": [s, w, n, e], "source": "OpenStreetMap via Overpass API"}
    cache.write_text(json.dumps(data), encoding="utf-8")
    log(f"[exposure] {len(elements)} OSM elements cached in {cache}")
    return data


def classify(el):
    tg = el.get("tags", {})
    if el["type"] == "node":
        if "place" in tg:
            return "settlement"
        return "facility"
    if tg.get("bridge") == "yes":
        return "bridge"
    if "building" in tg:
        return "facility" if tg.get("amenity") in ("school", "hospital", "clinic") else "building"
    if tg.get("power") == "plant":
        return "facility"
    if "highway" in tg:
        return "road"
    return "other"


def assets_local(sc):
    """OSM elements as sampled points in the local frame (roads resampled every 10 m for length)."""
    from pyproj import Transformer
    meta = sc.load_meta()
    fr = LocalFrame.from_dict(meta["frame"])
    tf = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    out = []
    for el in fetch_osm(sc)["elements"]:
        kind = classify(el)
        if kind == "other":
            continue
        if el["type"] == "node":
            pts = [(el["lon"], el["lat"])]
        else:
            pts = [(g["lon"], g["lat"]) for g in el.get("geometry", [])]
        if not pts:
            continue
        X, Y = tf.transform(*zip(*pts))
        x, y = fr.to_local(np.array(X), np.array(Y))
        if kind in ("road", "bridge") and len(x) > 1:
            seg = np.hypot(np.diff(x), np.diff(y))
            dense = [np.c_[np.linspace(x[i], x[i + 1], max(2, int(seg[i] / 10) + 1)),
                           np.linspace(y[i], y[i + 1], max(2, int(seg[i] / 10) + 1))] for i in range(len(seg))]
            P = np.vstack(dense)
            length = float(seg.sum())
        else:
            P = np.c_[[x.mean()], [y.mean()]]
            length = 0.0
        area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) if len(x) > 2 else 0.0
        tg = el.get("tags", {})
        out.append({"id": el["id"], "kind": kind, "name": tg.get("name:en") or tg.get("name", ""),
                    "tag": tg.get("highway") or tg.get("building") or tg.get("place") or tg.get("amenity") or "",
                    "pts": P, "length_m": length, "area_m2": area, "lonlat": pts[len(pts) // 2]})
    return out


# Huizinga, de Moel & Szewczyk (2017) JRC global flood depth-damage functions, Asia, residential buildings
HUIZINGA_DEPTH = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
HUIZINGA_ASIA_RES = np.array([0.00, 0.33, 0.49, 0.62, 0.72, 0.87, 0.93, 0.98, 1.00])


def building_damage(h, v):
    """Clausen & Clark (1990) dam-break damage class for masonry buildings + Huizinga (2017) damage ratio."""
    hv = h * v
    if v >= 2.0 and hv >= 7.0:
        return "destroyed", 1.0
    ratio = float(np.interp(h, HUIZINGA_DEPTH, HUIZINGA_ASIA_RES))
    if v >= 2.0 and hv >= 3.0:
        return "partially damaged", max(ratio, 0.5)
    return "inundated", ratio


def impacts(sc, store, fields, assets=None):
    """Assets reached by water (hmax >= wet threshold) and their max depth / velocity / arrival, per model."""
    meta = store.meta
    ag = meta["analysis_grid"]
    thr = sc["simulation"]["wet_threshold_m"]
    hmax = store.to_grid(fields["hmax"], 0.0)
    vmax = store.to_grid(fields["vmax"], 0.0)
    arr = store.to_grid(fields["arrival_min"])
    hz = store.to_grid(fields["hazard_class"], -1)
    resv = store.t["cell_reservoir"]
    assets = assets if assets is not None else assets_local(sc)
    hits = []
    summary = {"building": 0, "facility": 0, "settlement": 0, "bridge": 0, "road_km": 0.0, "road_impassable_km": 0.0,
               "damage": {"destroyed": 0, "partially damaged": 0, "inundated": 0, "footprint_m2": 0.0, "damage_equivalent_m2": 0.0,
                          "method": "Clausen & Clark (1990) dam-break criteria; Huizinga et al. (2017) JRC depth-damage, Asia residential"}}
    for a in assets:
        i = np.floor((a["pts"][:, 0] - ag["x0_corner"]) / ag["dx"]).astype(int)
        j = np.floor((a["pts"][:, 1] - ag["y0_corner"]) / ag["dx"]).astype(int)
        ok = (i >= 0) & (i < ag["nx"]) & (j >= 0) & (j < ag["ny"])
        if not ok.any():
            continue
        i, j = i[ok], j[ok]
        wet = (hmax[j, i] >= thr) & ~resv[j, i]
        if not wet.any():
            continue
        frac = wet.mean()
        rec = {"kind": a["kind"], "name": a["name"], "tag": a["tag"], "lon": a["lonlat"][0], "lat": a["lonlat"][1],
               "max_depth_m": float(hmax[j, i][wet].max()), "max_velocity_ms": float(vmax[j, i][wet].max()),
               "arrival_min": float(np.nanmin(arr[j, i][wet])) if np.isfinite(arr[j, i][wet]).any() else None,
               "hazard_class": int(hz[j, i][wet].max())}
        if a["kind"] == "road":
            rec["flooded_length_m"] = a["length_m"] * frac
            summary["road_km"] += a["length_m"] * frac / 1000.0
            summary["road_impassable_km"] += a["length_m"] * float(((hmax[j, i] >= 0.3) & ~resv[j, i]).mean()) / 1000.0
        else:
            summary[a["kind"]] += 1
        if a["kind"] in ("building", "facility"):
            cls, ratio = building_damage(rec["max_depth_m"], rec["max_velocity_ms"])
            rec["damage_class"], rec["damage_ratio"] = cls, ratio
            dmg = summary["damage"]
            dmg[cls] += 1
            dmg["footprint_m2"] += a.get("area_m2", 0.0)
            dmg["damage_equivalent_m2"] += a.get("area_m2", 0.0) * ratio
        hits.append(rec)
    summary["road_km"] = round(summary["road_km"], 2)
    summary["road_impassable_km"] = round(summary["road_impassable_km"], 2)
    summary["damage"]["footprint_m2"] = round(summary["damage"]["footprint_m2"])
    summary["damage"]["damage_equivalent_m2"] = round(summary["damage"]["damage_equivalent_m2"])
    return summary, hits
