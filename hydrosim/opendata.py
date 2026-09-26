"""
Open-data inputs for new scenarios (no accounts or keys needed):

  * Copernicus GLO-30 DEM tiles (ESA/Airbus, AWS open data bucket copernicus-dem-30m)
  * Sentinel-2 L2A true-colour imagery (Element 84 Earth Search STAC, AWS open data)
  * dams and weirs from OpenStreetMap (Overpass API), place search via Nominatim
"""
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

UA = {"User-Agent": "JalPravah/1.0 (dam-break research tool)"}
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
STAC = "https://earth-search.aws.element84.com/v1/search"
OVERPASS = "https://overpass-api.de/api/interpreter"
NOMINATIM = "https://nominatim.openstreetmap.org/search"


def _get(url, data=None, headers=None, timeout=120):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def dem_tile_name(lat, lon):
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def download_dem(bbox, out_path, log=print):
    """Mosaic of Copernicus GLO-30 tiles covering bbox (lon_min, lat_min, lon_max, lat_max), clipped to it."""
    import rasterio
    from rasterio.merge import merge
    lo0, la0, lo1, la1 = bbox
    srcs, paths = [], []
    for lat in range(math.floor(la0), math.floor(la1) + 1):
        for lon in range(math.floor(lo0), math.floor(lo1) + 1):
            n = dem_tile_name(lat, lon)
            try:
                srcs.append(rasterio.open(f"/vsicurl/{DEM_URL.format(n=n)}"))
                paths.append(n)
            except Exception:
                log(f"[dem] no Copernicus tile {n} (sea or outside coverage)")
    if not srcs:
        raise RuntimeError("No Copernicus GLO-30 tiles cover this area")
    log(f"[dem] mosaicking {len(srcs)} Copernicus GLO-30 tile(s): {', '.join(paths)}")
    arr, tr = merge(srcs, bounds=bbox, nodata=0)
    prof = srcs[0].profile.copy()
    prof.update(driver="GTiff", height=arr.shape[1], width=arr.shape[2], transform=tr, count=1, compress="deflate",
                tiled=True, blockxsize=256, blockysize=256, nodata=0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr[0], 1)
    for s in srcs:
        s.close()
    return {"source": "Copernicus GLO-30 (AWS open data)", "tiles": paths, "shape": list(arr.shape[1:])}


def download_sentinel2(bbox, out_path, months_back=12, max_cloud=10, log=print):
    """Least-cloudy recent Sentinel-2 L2A true-colour (TCI) scene over bbox, resampled to ~20 m, as RGB GeoTIFF."""
    import datetime as dt
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    end = dt.date.today()
    start = end - dt.timedelta(days=30 * months_back)
    q = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox), "limit": 50,
         "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "query": {"eo:cloud_cover": {"lt": max_cloud}}}
    items = json.loads(_get(STAC, json.dumps(q).encode(), {"Content-Type": "application/json"}))["features"]
    if not items:
        raise RuntimeError("No Sentinel-2 scene below the cloud limit in the last year")
    # prefer the scene whose footprint covers most of the bbox, then the least cloud
    from shapely.geometry import box, shape
    aoi = box(*bbox)
    items.sort(key=lambda f: (-shape(f["geometry"]).intersection(aoi).area / aoi.area, f["properties"]["eo:cloud_cover"]))
    it = items[0]
    href = it["assets"]["visual"]["href"]
    log(f"[s2] using {it['id']} ({it['properties']['datetime'][:10]}, cloud {it['properties']['eo:cloud_cover']:.1f} %)")
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        scale = 2  # 10 m -> 20 m
        h, w = max(1, int(win.height // scale)), max(1, int(win.width // scale))
        data = src.read([1, 2, 3], window=win, out_shape=(3, h, w), resampling=Resampling.average, boundless=True, fill_value=0)
        tr = src.window_transform(win) * rasterio.Affine.scale(win.width / w, win.height / h)
        prof = {"driver": "GTiff", "height": h, "width": w, "count": 3, "dtype": "uint8", "crs": src.crs,
                "transform": tr, "compress": "deflate", "photometric": "RGB"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(data.astype(np.uint8))
    return {"source": "Sentinel-2 L2A true colour (Earth Search, AWS open data)", "scene": it["id"],
            "date": it["properties"]["datetime"][:10], "cloud_pct": it["properties"]["eo:cloud_cover"]}


def find_dams(bbox):
    """OpenStreetMap dams/weirs in bbox (lon_min, lat_min, lon_max, lat_max)."""
    lo0, la0, lo1, la1 = bbox
    b = f"{la0},{lo0},{la1},{lo1}"
    q = (f'[out:json][timeout:60];(way["waterway"~"^(dam|weir)$"]({b});node["waterway"~"^(dam|weir)$"]({b});'
         f'way["man_made"="dam"]({b});relation["waterway"="dam"]({b}););out center tags;')
    els = None
    for attempt in range(3):
        for url in (OVERPASS, "https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter"):
            try:
                els = json.loads(_get(url, urllib.parse.urlencode({"data": q}).encode(), timeout=90))["elements"]
                break
            except Exception:
                continue
        if els is not None:
            break
        __import__("time").sleep(3 * (attempt + 1))
    if els is None:
        raise RuntimeError("OpenStreetMap servers are busy; try again in a minute")
    out = []
    for e in els:
        c = e.get("center", {"lat": e.get("lat"), "lon": e.get("lon")})
        if c.get("lat") is None:
            continue
        tg = e.get("tags", {})
        out.append({"osm": f"{e['type']}/{e['id']}", "lat": c["lat"], "lon": c["lon"],
                    "name": tg.get("name:en") or tg.get("name") or "", "kind": tg.get("waterway") or tg.get("man_made"),
                    "height": tg.get("height", "")})
    return out


def geocode(query, limit=6):
    url = NOMINATIM + "?" + urllib.parse.urlencode({"q": query, "format": "json", "limit": limit})
    return [{"name": r["display_name"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in json.loads(_get(url))]
