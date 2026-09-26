# hydrosim — DEM-driven dam-break simulation with SPH and Delft3D

Two independent hydrodynamic models are run on one physically identical problem that is derived
entirely from the DEM, then compared cell by cell.

| | SPH | Delft3D |
|---|---|---|
| Engine | DualSPHysics 5.4 CPU (OpenMP), compiled from `DualSPHysics-master/src` | Delft3D-FLOW 4, tag 5.01.00.2163, compiled from the Deltares source in `deltares-delft3d-b-master (3)` |
| Physics | 3-D weakly compressible Navier–Stokes, Wendland kernel, artificial viscosity, Fourtakas density diffusion, DBC boundaries | Depth-averaged shallow-water equations, Manning friction, "Flood" advection scheme, wetting/drying |
| Discretisation | particles, `sph.dp_m` spacing | 20 m cells (`delft3d.cell_size_m`) |
| Runs in | Docker image `hydrosim/dualsphysics:5.4-run` | Docker image `hydrosim/delft3d-flow:5.01-run` |

No machine-learning surrogate is used anywhere in this pipeline. The HydroUNet model in
`dam_break_system/ml_surrogate` is a separate project and is untouched.

## Scenario (config/trishuli_scenario.json)

* DEM: `Nepal_Data/dem.tif` (30 m), reprojected to UTM 45N.
* River path: priority-flood depression filling + D8 flow routing; the dam is placed at the narrowest valley
  section within 1.5 km downstream of the given point (Trishuli near Betrawati).
* Dam: 55 m embankment (20 m crest, 2:1 slopes) burned into the terrain.
* Reservoir: every DEM cell below the crest that is hydraulically connected upstream of the dam
  (≈30 Mm³ for this site).
* Breach: Froehlich (2008) average width and side slope for that volume and height, opened instantaneously.
* Domain: valley corridor within 45 m of the river bed, 30 km downstream; free outfall at the end.


## Problem-statement deliverables in JalPravah

| PS item | Where | How |
|---|---|---|
| i. Dam-break / river-blockage modelling with SPH and Delft3D, loss & damage | every scenario page; `hydrosim/sph_model.py`, `delft3d_model.py`, `exposure.py` | DualSPHysics and Delft3D-FLOW run independently on the same DEM-derived dam, reservoir and Froehlich breach. Dam type "Landslide dam / river blockage" uses a wide-crested, flat-faced blockage. Damage: Clausen & Clark (1990) dam-break criteria + Huizinga et al. (2017) JRC depth-damage (Asia) on OpenStreetMap building footprints; roads flooded / impassable; bridges, places, schools, health facilities |
| ii. Scenarios from different input datasets | **New scenario** page; `hydrosim/builder.py`, `opendata.py` | any location: Copernicus GLO-30 DEM (auto) or your GeoTIFF; Sentinel-2 imagery (auto), your image or none; OSM dam search or map click; dam height/type; reach, duration, cell size, SPH spacing, Manning n |
| iii. Dashboard for inputs and outputs, large data, .shp/.kml | whole app | 2-D and 3-D animated results, comparison, downloads of GeoTIFF, zipped shapefiles, KML, GeoJSON and raw solver output (GB-scale) |
| iv. Near-real-time flood analysis on Google Earth Engine | **Satellite flood mapping** page; `hydrosim/gee.py` | Sentinel-1 VV change detection with JRC permanent-water and GLO-30 slope masks; results as .shp/.kml/.geojson and overlaid on the modelled extents. Needs your Google account with Earth Engine access and a Cloud project; nothing is shown without it |
| v. Indian river and dam | scenario **Maneri dam, Bhagirathi (Uttarakhand)** | built entirely from open data (OSM dam location, Copernicus DEM, Sentinel-2) |

Start the app with `JalPravah.bat` (or `python jalpravah.py`) and open http://localhost:8765/.
Earth Engine: open *Satellite flood mapping*, press *Sign in with Google*, then enter the Cloud project ID registered for Earth Engine.

## Running

Docker Desktop must be running, with its disk image on D:.

```
cd hydrosim
python pipeline.py scenario      # DEM -> terrain.npz, scenario_meta.json
python pipeline.py delft3d       # ~3 h for 3 h of flood
python pipeline.py sph           # ~20 h for 90 min of flood at dp = 12 m on 10 cores
python pipeline.py analyze       # extraction, metrics, GIS, OSM exposure, comparison, viewer
```

The SPH solver runs in a container named `hydrosim_sph_solver`; its snapshots stay in the Docker volume
`hydrosim_<scenario>_sph` (`/work/case/out/data`). Progress: `Run.out` in that folder.

## Outputs (runs/<scenario>/)

* `scenario/`: terrain grid, reservoir/dam/breach geometry, stations, cached OSM exposure
* `delft3d/case/`, `sph/case/`: complete solver inputs and logs (re-runnable)
* `results/<model>/`: `frames.npz` (depth, u, v every 30 s), `series.json`, `metrics.json`, `max_fields.npz`,
  `impacts.json`, `gis/` (GeoTIFF, .shp, .kml, .geojson)
* `results/comparison.json`: map agreement and station-by-station differences
* `viewer/`: `sph.html`, `delft3d.html`, `comparison.html` (open directly in a browser)

## Known limits

* 3-D SPH at river scale is expensive: on a 6-core laptop the finest affordable spacing for the whole event
  is 12 m, so a shallow flood layer is only one or two particles deep. The near field (breach, gorge) is
  resolved far better than thin floodplain edges.
* SPH has no bed-friction law (energy is lost through artificial viscosity and the boundary); Delft3D uses
  Manning n = 0.045. Differences in attenuation partly come from this.
* The breach is instantaneous in both models: Delft3D-FLOW 4 has no time-varying crest structure, so a
  gradual breach could not be represented identically in the two engines.

## JalPravah on the web (Vercel)

https://jalpravah-geoint.vercel.app is the same app as http://localhost:8765/.

* `python export_web.py` writes `web/`: the app pages, a status / sign-in bar, a Vercel function
  (`api/proxy.js`) and the finished scenarios as static files.
* `JalPravah Online.bat` (`python jalpravah_online.py`) connects the site to this workstation: it opens a
  Cloudflare quick tunnel to the local server, stores the tunnel URL, a shared key and the password in the
  Vercel project, and redeploys. While that window is open, New scenario, Run / Re-run, Delete and Satellite
  flood mapping on the website run here, on Docker and your Earth Engine sign-in, exactly as locally.
* Reading is public; actions need the password (printed by the launcher, kept in `.jalpravah_remote.json`,
  never committed). The local server answers tunnelled requests only when they carry the shared key.
* With the window closed the site shows the saved results and says the workstation is offline.
