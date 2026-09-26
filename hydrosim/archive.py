"""
Saves every generated input, mesh and raw solver output of a scenario to D: in reusable formats:

runs/<scenario>/
  scenario/gis/      terrain_with_dam_utm.tif, dem_original_utm.tif, reservoir/dam/breach/domain/stations/river (.shp .geojson .kml)
  delft3d/case/      complete Delft3D-FLOW model: .mdf .grd .enc .dep .dry .ini .bnd .bct .obs .crs (re-runnable)
  delft3d/output/    raw NEFIS results trim-dam.dat/.def (maps), trih-dam.dat/.def (stations), tri-diag.dam
  sph/case/          complete DualSPHysics case: Dambreak_Def.xml, terrain.stl, gate.stl
  sph/output/        GenCase particles (Dambreak.bi4/.xml), Part_XXXX.bi4 snapshots, Run.out, Run.csv, fluid VTK
  manifest.json      every file with size and description
"""
import json
import math
from pathlib import Path

import numpy as np

from .terrain import LocalFrame, load_terrain


def _run(args):
    import os
    import subprocess
    return subprocess.run(args, env=dict(os.environ, MSYS_NO_PATHCONV="1"), capture_output=True, text=True)


def copy_from_volume(volume, src, dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    host = str(dst.resolve()).replace("\\", "/")
    r = _run(["docker", "run", "--rm", "-v", f"{volume}:/work:ro", "-v", f"{host}:/dst", "alpine", "sh", "-c",
              f"cp -rf {src} /dst/ 2>&1; ls /dst | wc -l"])
    return r.returncode, r.stdout + r.stderr


def archive_solver_outputs(sc, model):
    if model == "delft3d":
        vol = f"hydrosim_{sc.name}_d3d"
        out = sc.delft3d_dir / "output"
        return copy_from_volume(vol, "/work/case/trim-dam.* /work/case/trih-dam.* /work/case/tri-diag.dam", out)
    vol = f"hydrosim_{sc.name}_sph"
    out = sc.sph_dir / "output"
    return copy_from_volume(vol, "/work/case/out/.", out)


def export_scenario_gis(sc):
    """Scenario geometry as north-up UTM rasters and vector layers (shapefile, GeoJSON, KML)."""
    import rasterio
    import shapefile
    import simplekml
    from pyproj import CRS, Transformer
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from shapely.geometry import LineString, mapping, shape
    from shapely.ops import transform as stransform
    from shapely.ops import unary_union

    t, meta = load_terrain(sc), sc.load_meta()
    fr = LocalFrame.from_dict(meta["frame"])
    out = sc.scenario_dir / "gis"
    out.mkdir(parents=True, exist_ok=True)
    res = meta["resolution_m"]
    xs, ys = t["xs"], t["ys"]
    X, Y = fr.to_utm(np.array([xs[0], xs[-1], xs[0], xs[-1]]), np.array([ys[0], ys[0], ys[-1], ys[-1]]))
    xmin, xmax, ymin, ymax = X.min(), X.max(), Y.min(), Y.max()
    nx, ny = int(math.ceil((xmax - xmin) / res)), int(math.ceil((ymax - ymin) / res))
    tr = from_origin(xmin, ymax, res, res)
    UX, UY = np.meshgrid(xmin + (np.arange(nx) + 0.5) * res, ymax - (np.arange(ny) + 0.5) * res)
    lx, ly = fr.to_local(UX, UY)
    ii = np.round((lx - xs[0]) / res).astype(int)
    jj = np.round((ly - ys[0]) / res).astype(int)
    ok = (ii >= 0) & (ii < len(xs)) & (jj >= 0) & (jj < len(ys))

    def raster(a, dtype="float32", nodata=np.nan):
        r = np.full((ny, nx), nodata, dtype)
        r[ok] = a[jj[ok], ii[ok]]
        return r

    for name, arr in (("terrain_with_dam_utm", t["z"]), ("dem_original_utm", t["z_orig"])):
        with rasterio.open(out / f"{name}.tif", "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float32",
                           crs=meta["crs"], transform=tr, nodata=np.nan, compress="deflate") as ds:
            ds.write(raster(arr.astype(np.float32)), 1)

    to_wgs = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)
    wkt = CRS.from_epsg(4326).to_wkt(version="WKT1_ESRI")

    def poly(mask):
        m = raster(mask.astype(np.uint8), "uint8", 0)
        g = [shape(p) for p, v in shapes(m, mask=m == 1, transform=tr) if v == 1]
        return unary_union(g).simplify(res * 0.5) if g else None

    def local_line(pts):
        U = fr.to_utm(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
        return LineString(list(zip(*U)))

    layers = {
        "reservoir": [(poly(t["reservoir"]), {"name": "Reservoir at crest level", "value": round(meta["reservoir"]["volume_m3"] / 1e6, 3), "unit": "Mm3"})],
        "dam": [(poly(t["dam"]), {"name": "Dam embankment", "value": meta["dam"]["height_m"], "unit": "m high"})],
        "breach": [(poly(t["notch"]), {"name": "Breach notch (Froehlich 2008)", "value": round(meta["breach"]["bottom_width_m"], 1), "unit": "m base width"})],
        "domain": [(poly(t["domain"]), {"name": "Simulation domain", "value": round(meta["domain_area_m2"] / 1e6, 2), "unit": "km2"})],
        "stations": [(local_line(s["section"]), {"name": s["name"], "value": round(s["chainage_m"] / 1000, 2), "unit": "km"}) for s in meta["stations"]],
        "river_thalweg": [(local_line(t["thalweg"][:, :2].tolist()), {"name": "Main river (D8 flow path)", "value": 0, "unit": ""})],
    }
    for lname, items in layers.items():
        items = [(stransform(lambda x, y, z=None: to_wgs.transform(x, y), g), p) for g, p in items if g is not None]
        is_line = items and items[0][0].geom_type in ("LineString", "MultiLineString")
        w = shapefile.Writer(str(out / lname), shapeType=shapefile.POLYLINE if is_line else shapefile.POLYGON)
        for f, typ, size in (("name", "C", 60), ("value", "N", 14), ("unit", "C", 20)):
            w.field(f, typ, size, 3 if typ == "N" else 0)
        kml = simplekml.Kml(name=lname)
        feats = []
        for g, p in items:
            geoms = list(g.geoms) if hasattr(g, "geoms") else [g]
            if is_line:
                w.line([list(gg.coords) for gg in geoms])
                for gg in geoms:
                    kml.newlinestring(name=p["name"], coords=list(gg.coords)).style.linestyle.width = 3
            else:
                from shapely.geometry.polygon import orient
                rings = []
                for gg in geoms:
                    gg = orient(gg, -1.0)
                    rings.append(list(gg.exterior.coords))
                    rings += [list(r.coords) for r in gg.interiors]
                    kml.newpolygon(name=p["name"], outerboundaryis=list(gg.exterior.coords),
                                   innerboundaryis=[list(r.coords) for r in gg.interiors]).style.polystyle.color = "7fff7f00"
                w.poly(rings)
            w.record(p["name"], p["value"], p["unit"])
            feats.append({"type": "Feature", "geometry": mapping(g), "properties": p})
        w.close()
        (out / f"{lname}.prj").write_text(wkt)
        kml.save(str(out / f"{lname}.kml"))
        (out / f"{lname}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return out


DESCRIPTIONS = {
    ".mdf": "Delft3D-FLOW master definition file", ".grd": "Delft3D curvilinear grid (mesh) corners",
    ".enc": "Delft3D grid enclosure", ".dep": "Delft3D bathymetry (depth, positive down)", ".dry": "Delft3D dry points",
    ".ini": "Delft3D initial water level / velocity", ".bnd": "Delft3D open boundary", ".bct": "Delft3D boundary time series",
    ".obs": "Delft3D observation points", ".crs": "Delft3D cross-sections", ".stl": "SPH geometry mesh (STL)",
    ".bi4": "DualSPHysics particle data", ".tif": "GeoTIFF raster", ".shp": "ESRI shapefile", ".kml": "Google Earth KML",
    ".geojson": "GeoJSON", ".npz": "NumPy archive", ".json": "JSON", ".xml": "XML definition", ".vtk": "VTK particles",
    ".dat": "Delft3D NEFIS data", ".def": "Delft3D NEFIS definition", ".out": "solver log", ".csv": "CSV table",
}


def write_manifest(sc):
    root = sc.run_dir
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and "dumps" not in p.parts and "viewer" not in p.parts and "csv" not in p.parts:
            files.append({"path": str(p.relative_to(root)).replace("\\", "/"), "bytes": p.stat().st_size,
                          "description": DESCRIPTIONS.get(p.suffix.lower(), "")})
    man = {"scenario": sc.name, "root": str(root), "files": files,
           "total_gb": round(sum(f["bytes"] for f in files) / 1e9, 3)}
    (root / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man
