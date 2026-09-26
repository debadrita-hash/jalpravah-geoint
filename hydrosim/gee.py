"""
Near-real-time flood mapping on Google Earth Engine with open Sentinel-1 SAR data (PS deliverable iv).

Method (change detection, e.g. UN-SPIDER recommended practice for Sentinel-1 flood mapping):
  1. Sentinel-1 GRD, IW mode, VV polarisation, same orbit direction for both windows
  2. pre-event reference = median backscatter of the "before" window, post-event = minimum of the "after" window
  3. 50 m focal-mean speckle smoothing, difference in dB
  4. flooded where the drop exceeds the threshold (default 3 dB) and post-event VV < -15 dB
  5. masks: permanent water (JRC Global Surface Water occurrence > 80 %), slopes > 5 degrees (Copernicus GLO-30),
     patches smaller than 8 connected pixels
  6. vectorised at 20 m and returned with the flooded area and the scenes used

Needs a Google account with Earth Engine access and a Cloud project; nothing is computed without it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import HYDROSIM_ROOT

SETTINGS = HYDROSIM_ROOT / "config" / "gee_settings.json"
OUT_ROOT = HYDROSIM_ROOT / "runs" / "_satellite"


def settings():
    return json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}


def save_project(project):
    s = settings()
    s["project"] = project.strip()
    SETTINGS.write_text(json.dumps(s, indent=2))


def _init():
    import ee
    project = settings().get("project")
    if not project:
        raise RuntimeError("Enter your Google Cloud project ID (registered for Earth Engine) first.")
    ee.Initialize(project=project)
    return ee


def status():
    try:
        import ee  # noqa: F401
    except ImportError:
        return {"ready": False, "step": "install", "message": "The earthengine-api package is not installed."}
    cred = Path(os.path.expanduser("~")) / ".config" / "earthengine" / "credentials"
    if not cred.exists():
        return {"ready": False, "step": "authenticate", "message": "Sign in to Google Earth Engine to enable satellite flood mapping."}
    if not settings().get("project"):
        return {"ready": False, "step": "project", "message": "Enter the Google Cloud project registered for Earth Engine."}
    try:
        _init()
        return {"ready": True, "project": settings()["project"]}
    except Exception as ex:
        return {"ready": False, "step": "project", "message": f"Earth Engine refused the connection: {str(ex)[:300]}"}


def authenticate():
    """Opens Google's sign-in page in the browser (local machine) and stores the credentials."""
    return subprocess.Popen([sys.executable, "-c", "import ee; ee.Authenticate(auth_mode='localhost', force=True)"])


def map_flood(bbox, pre, post, threshold_db=3.0, orbit="DESCENDING", label="aoi"):
    """bbox = [lon_min, lat_min, lon_max, lat_max]; pre/post = [start, end] ISO dates."""
    ee = _init()
    aoi = ee.Geometry.Rectangle(bbox)
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.eq("orbitProperties_pass", orbit)).select("VV"))
    before, after = s1.filterDate(pre[0], pre[1]), s1.filterDate(post[0], post[1])
    n_before, n_after = before.size().getInfo(), after.size().getInfo()
    if n_before == 0 or n_after == 0:
        raise RuntimeError(f"No Sentinel-1 {orbit.lower()} scenes: {n_before} before, {n_after} after the event. "
                           "Widen the date windows or switch the orbit direction.")
    smooth = lambda img: img.focal_mean(50, "circle", "meters")
    ref = smooth(before.median()).clip(aoi)
    evt = smooth(after.min()).clip(aoi)
    diff = evt.subtract(ref)
    permanent = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gt(80).unmask(0)
    slope = ee.Terrain.slope(ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().setDefaultProjection("EPSG:4326", None, 30))
    flood = diff.lt(-abs(threshold_db)).And(evt.lt(-15)).And(permanent.Not()).And(slope.lt(5))
    flood = flood.updateMask(flood).updateMask(flood.connectedPixelCount(25, True).gte(8)).rename("flood")
    area_km2 = (flood.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), aoi, 20, maxPixels=1e10)
                .get("flood").getInfo() or 0.0) / 1e6
    vec = flood.reduceToVectors(geometry=aoi, scale=20, geometryType="polygon", eightConnected=True, maxPixels=1e10,
                                bestEffort=True)
    gj = vec.getInfo()
    dates = lambda c: sorted(set(c.aggregate_array("system:time_start").map(
        lambda t: ee.Date(t).format("YYYY-MM-dd")).getInfo()))
    info = {"aoi": bbox, "orbit": orbit, "threshold_db": threshold_db, "pre_window": pre, "post_window": post,
            "scenes_before": n_before, "scenes_after": n_after, "dates_before": dates(before), "dates_after": dates(after),
            "flooded_area_km2": round(area_km2, 3), "polygons": len(gj.get("features", [])),
            "method": "Sentinel-1 VV change detection; JRC GSW permanent water and GLO-30 slope masks"}
    return info, gj


def save(info, gj, run_id):
    """GeoJSON + shapefile + KML of the satellite flood extent."""
    import shapefile
    import simplekml
    from shapely.geometry import shape
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "flood_extent.geojson").write_text(json.dumps(gj))
    (out / "info.json").write_text(json.dumps(info, indent=2))
    w = shapefile.Writer(str(out / "flood_extent"), shapeType=shapefile.POLYGON)
    w.field("source", "C", 40)
    w.field("area_m2", "N", 14, 1)
    kml = simplekml.Kml(name=f"Sentinel-1 flood extent {run_id}")
    from shapely.geometry.polygon import orient
    for f in gj.get("features", []):
        g = shape(f["geometry"])
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        rings = []
        for pg in polys:
            pg = orient(pg, -1.0)
            rings.append([list(c) for c in pg.exterior.coords])
            rings += [[list(c) for c in r.coords] for r in pg.interiors]
            k = kml.newpolygon(name="Flooded (Sentinel-1)", outerboundaryis=list(pg.exterior.coords),
                               innerboundaryis=[list(r.coords) for r in pg.interiors])
            k.style.polystyle.color = "7fff5500"
        w.poly(rings)
        w.record("Sentinel-1 change detection", 0.0)
    w.close()
    from pyproj import CRS
    (out / "flood_extent.prj").write_text(CRS.from_epsg(4326).to_wkt(version="WKT1_ESRI"))
    kml.save(str(out / "flood_extent.kml"))
    return out


if __name__ == "__main__":
    # python -m hydrosim.gee <params.json>  ->  runs/_satellite/<run_id>/{flood_extent.geojson,.shp,.kml,info.json}
    p = json.loads(Path(sys.argv[1]).read_text())
    info, gj = map_flood(p["bbox"], p["pre"], p["post"], float(p.get("threshold_db", 3.0)), p.get("orbit", "DESCENDING"))
    info["label"] = p.get("label", "")
    out = save(info, gj, p["run_id"])
    print(f"[satellite] {info['flooded_area_km2']} km2 flooded, {info['polygons']} polygons -> {out}", flush=True)
