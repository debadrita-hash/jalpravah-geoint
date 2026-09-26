"""
SPH vs Delft3D comparison, computed only from the two independent result products.

Map agreement follows the flood-mapping literature (e.g. Bates & De Roo 2000; Wing et al. 2017):
    CSI = A / (A + B + C), hit rate = A / (A + C), false alarm ratio = B / (A + B), bias = (A + B) / (A + C)
with A = wet in both, B = wet in SPH only, C = wet in Delft3D only (Delft3D taken as reference).
"""
import json

import numpy as np

from .results import ResultStore


def _load(store, suffix="_window"):
    m = json.loads((store.dir / f"metrics{suffix}.json").read_text(encoding="utf-8"))
    f = np.load(store.dir / f"max_fields{suffix}.npz")
    return m, {k: f[k] for k in f.files}


def compare(sc):
    """Both models are evaluated over the common window [0, min(T_sph, T_delft3d)]."""
    from .results import compute_metrics
    sph, d3d = ResultStore(sc, "sph"), ResultStore(sc, "delft3d")
    t_common = min(float(np.load(st.dir / "frames.npz")["times"][-1]) for st in (sph, d3d))
    for st in (sph, d3d):
        compute_metrics(st, t_max=t_common, suffix="_window")
    ms, fs = _load(sph)
    md, fd = _load(d3d)
    thr = sc["simulation"]["wet_threshold_m"]
    resv = sph.t["cell_reservoir"].ravel()[sph.idx]
    ws, wd = (fs["hmax"] >= thr) & ~resv, (fd["hmax"] >= thr) & ~resv
    A, B, C = int((ws & wd).sum()), int((ws & ~wd).sum()), int((~ws & wd).sum())
    area = sph.meta["analysis_grid"]["dx"] ** 2 / 1e6
    union = ws | wd
    both = ws & wd
    dh = fs["hmax"][union] - fd["hmax"][union]
    dv = fs["vmax"][both] - fd["vmax"][both]
    arr_s, arr_d = fs["arrival_min"][both], fd["arrival_min"][both]
    ok = np.isfinite(arr_s) & np.isfinite(arr_d)
    corr = float(np.corrcoef(fs["hmax"][both], fd["hmax"][both])[0, 1]) if both.sum() > 2 else None

    agreement = {
        "wet_threshold_m": thr,
        "area_both_km2": A * area, "area_sph_only_km2": B * area, "area_delft3d_only_km2": C * area,
        "critical_success_index": A / max(A + B + C, 1),
        "hit_rate": A / max(A + C, 1), "false_alarm_ratio": B / max(A + B, 1), "area_bias": (A + B) / max(A + C, 1),
        "hmax_mean_difference_m": float(dh.mean()) if len(dh) else None,
        "hmax_rmse_m": float(np.sqrt((dh ** 2).mean())) if len(dh) else None,
        "hmax_correlation": corr,
        "vmax_mean_difference_ms": float(dv.mean()) if len(dv) else None,
        "arrival_mean_difference_min": float((arr_s[ok] - arr_d[ok]).mean()) if ok.any() else None,
    }

    def rel(a, b):
        return None if a is None or b in (None, 0) else 100.0 * (a - b) / abs(b)

    # the comparable metric set shown everywhere in JalPravah (same definitions, same grid, same window)
    from .exposure import assets_local, impacts
    assets = assets_local(sc)
    exp_s = impacts(sc, sph, fs, assets)[0]
    exp_d = impacts(sc, d3d, fd, assets)[0]

    def st_val(m, km, key):
        s = min(m["stations"], key=lambda x: abs(x["chainage_km"] - km))
        return s.get(key)

    spec = [
        ("breach_peak_discharge_m3s", "Peak discharge at the breach", "m³/s", lambda m, e: m["breach_peak_discharge_m3s"], 0),
        ("breach_time_to_peak_min", "Time to breach peak", "min", lambda m, e: m["breach_time_to_peak_min"], 1),
        ("flooded_area_km2", "Flooded area", "km²", lambda m, e: m["flooded_area_km2"], 2),
        ("max_depth_m", "Maximum flow depth", "m", lambda m, e: m["max_depth_m"], 1),
        ("max_velocity_ms", "Maximum velocity", "m/s", lambda m, e: m["max_velocity_ms"], 1),
        ("max_front_distance_km", "Flood front reach", "km", lambda m, e: m["max_front_distance_km"], 1),
        ("arrival_10km_min", "Arrival at 10 km", "min", lambda m, e: st_val(m, 10, "arrival_time_min"), 1),
        ("arrival_25km_min", "Arrival at 25 km", "min", lambda m, e: st_val(m, 25, "arrival_time_min"), 1),
        ("peak_q_10km_m3s", "Peak discharge at 10 km", "m³/s", lambda m, e: st_val(m, 10, "peak_discharge_m3s") or None, 0),
        ("buildings", "Buildings reached", "count", lambda m, e: e["building"], 0),
        ("buildings_destroyed", "Buildings destroyed (Clausen & Clark)", "count", lambda m, e: e.get("damage", {}).get("destroyed"), 0),
        ("roads_km", "Roads flooded", "km", lambda m, e: e["road_km"], 1),
        ("extreme_hazard_km2", "Area at extreme hazard", "km²", lambda m, e: m["hazard_area_km2"]["Extreme - danger for all"], 2),
    ]
    rows = []
    for key, label, unit, fn, dec in spec:
        a, b = fn(ms, exp_s), fn(md, exp_d)
        a = None if a is None else float(a)
        b = None if b is None else float(b)
        rows.append({"key": key, "label": label, "unit": unit, "decimals": dec, "sph": a, "delft3d": b,
                     "difference": None if a is None or b is None else a - b, "relative_pct": rel(a, b)})

    stations = []
    for s_s, s_d in zip(ms["stations"], md["stations"]):
        st = {"name": s_s["name"], "chainage_km": s_s["chainage_km"]}
        for k in ("peak_discharge_m3s", "time_of_peak_discharge_min", "peak_depth_m", "arrival_time_min",
                  "passed_volume_Mm3"):
            st[k] = {"sph": s_s.get(k), "delft3d": s_d.get(k),
                     "difference": None if s_s.get(k) is None or s_d.get(k) is None else s_s[k] - s_d[k]}
        stations.append(st)

    result_reading = _reading(sc, agreement, rows, stations, ms, md)
    result = {"agreement": agreement, "summary_rows": rows, "stations": stations, "reading": result_reading,
              "run": {"sph": ms["run"], "delft3d": md["run"]},
              "mass_balance": {"sph": ms["mass_balance"], "delft3d": md["mass_balance"]},
              "common_window_min": min(ms["timeseries"]["time_min"][-1], md["timeseries"]["time_min"][-1])}
    np.savez_compressed(sc.results_dir / "comparison_fields.npz",
                        agreement=(ws & wd) * 1 + (ws & ~wd) * 2 + (~ws & wd) * 3,
                        hmax_difference=np.where(union, fs["hmax"] - fd["hmax"], np.nan).astype(np.float32))
    (sc.results_dir / "comparison.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _reading(sc, g, rows, stations, ms, md):
    """Plain-language notes generated from the computed numbers (no fixed conclusions)."""
    r = {x["key"]: x for x in rows}
    f = lambda v, d=0: "–" if v is None else f"{v:,.{d}f}"
    out = []
    q = r["breach_peak_discharge_m3s"]
    if q["sph"] is not None and q["delft3d"]:
        more = "higher" if q["sph"] > q["delft3d"] else "lower"
        out.append(f"Peak breach discharge is {f(q['sph'])} m³/s in SPH and {f(q['delft3d'])} m³/s in Delft3D "
                   f"({f(abs(q['relative_pct']), 0)} % {more} in SPH). SPH resolves the vertical acceleration through the "
                   f"notch in 3-D; Delft3D assumes hydrostatic pressure in a single depth-averaged layer.")
    s_last = [s for s in stations if s["peak_discharge_m3s"]["sph"] and s["peak_discharge_m3s"]["delft3d"]]
    if len(s_last) >= 2:
        a, b = s_last[1], s_last[-1]
        att_s = 100 * (1 - b["peak_discharge_m3s"]["sph"] / a["peak_discharge_m3s"]["sph"])
        att_d = 100 * (1 - b["peak_discharge_m3s"]["delft3d"] / a["peak_discharge_m3s"]["delft3d"])
        out.append(f"Between {a['name']} and {b['name']} the flood peak attenuates by {f(att_s, 0)} % in SPH and "
                   f"{f(att_d, 0)} % in Delft3D. Delft3D dissipates energy through Manning bed friction "
                   f"(n = {sc['delft3d']['manning_n']}); SPH has no bed-friction law and loses energy only through "
                   f"artificial viscosity and the particle boundary.")
    arr = [s for s in stations if s["arrival_time_min"]["difference"] is not None]
    if arr:
        far = arr[-1]
        lead = "earlier" if far["arrival_time_min"]["difference"] < 0 else "later"
        out.append(f"At {far['name']} the flood arrives {f(abs(far['arrival_time_min']['difference']), 1)} min {lead} in SPH "
                   f"({f(far['arrival_time_min']['sph'], 1)} vs {f(far['arrival_time_min']['delft3d'], 1)} min).")
    out.append(f"The two maximum-extent maps give a critical success index of {g['critical_success_index']:.2f} "
               f"(1 = identical). {f(g['area_sph_only_km2'], 2)} km² is flooded only in SPH and "
               f"{f(g['area_delft3d_only_km2'], 2)} km² only in Delft3D.")
    out.append(f"Resolution differs by design: SPH uses {sc['sph']['dp_m']:g} m particles, the finest spacing this laptop can run "
               f"for the whole event, so a thin flood layer is only one or two particles deep, while Delft3D computes "
               f"on {sc['delft3d']['cell_size_m']:g} m cells. Shallow floodplain edges are therefore the least reliable "
               f"part of the SPH map.")
    return out
