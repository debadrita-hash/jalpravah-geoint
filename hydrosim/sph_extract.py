"""
DualSPHysics particles -> standard hydrosim result product (runs inside the SPH container).

    python3 /hydrosim/sph_extract.py <csv_dir> <scenario_dir> <out_dir> <tout_s> <dp> <coefh> <x_end>

For every PartVTK CSV snapshot (fluid particles: position, velocity, density, mass, id):
  * depth on each active 20 m analysis cell = column integral of the SPH kernel:
        h(x) = sum_j V_j W2D(|x - x_j|, h_sph)      (Wendland C2, V_j = m_j / rho_j)
    which is the SPH-consistent, mass-conserving flow depth; depth-averaged velocity is the
    kernel-weighted mean of the particle velocities
  * discharge through each station section = particle volume crossing the section line between
    consecutive snapshots (matched by particle id) / interval
  * particles past x_end = outflow; particles that vanish upstream of x_end = numerical loss
Only numpy/scipy are required.
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree


def read_csv(path):
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    hi = next(i for i, l in enumerate(lines) if "Pos.x" in l or "Pos.X" in l)
    sep = ";" if lines[hi].count(";") > lines[hi].count(",") else ","
    names = [c.strip().split(" ")[0].lower() for c in lines[hi].strip().split(sep)]
    rows = [l.strip().rstrip(sep) for l in lines[hi + 1:] if l.strip()]  # PartVTK ends each row with a separator
    data = np.loadtxt(rows, delimiter=sep, ndmin=2) if rows else np.zeros((0, len(names)))
    col = {n: data[:, k] for k, n in enumerate(names) if n}
    return col


def wendland2d(q, h):
    return np.where(q < 2.0, 7.0 / (4.0 * math.pi * h * h) * (1.0 - 0.5 * q) ** 4 * (2.0 * q + 1.0), 0.0)


def main(csv_dir, scen_dir, out_dir, tout, dp, coefh, x_end):
    tout, dp, coefh, x_end = float(tout), float(dp), float(coefh), float(x_end)
    h_sph = coefh * math.sqrt(3.0) * dp
    t = np.load(os.path.join(scen_dir, "terrain.npz"))
    with open(os.path.join(scen_dir, "scenario_meta.json")) as f:
        meta = json.load(f)
    act = t["cell_domain"]
    idx = np.nonzero(act.ravel())[0]
    XC, YC = np.meshgrid(t["xc"], t["yc"])
    cells = np.c_[XC.ravel()[idx], YC.ravel()[idx]]
    ctree = cKDTree(cells)
    files = sorted(glob.glob(os.path.join(csv_dir, "Fluid_[0-9][0-9][0-9][0-9].csv")))
    nt = len(files)
    depth = np.zeros((nt, len(idx)), np.float32)
    u = np.zeros_like(depth)
    v = np.zeros_like(depth)
    times = np.arange(nt) * tout
    stations = meta["stations"]
    q = np.zeros((len(stations), nt))
    out_cum = np.zeros(nt)
    lost_cum = np.zeros(nt)
    vol_in = np.zeros(nt)
    prev = None
    excluded_out = excluded_lost = 0.0
    X0P, Y0P, Z0P = float(t["xs"][0]) - 100.0, float(t["ys"][0]) - 100.0, float(np.nanmin(t["z"])) - 50.0
    p3d_t, p3d_xyz, p3d_s = [], [], []
    v0 = None
    for k, fpath in enumerate(files):
        c = read_csv(fpath)
        x, y, z = c["pos.x"], c["pos.y"], c["pos.z"]
        vx, vy = c["vel.x"], c["vel.y"]
        vol = c["mass"] / c["rhop"]
        ids = c["idp"].astype(np.int64)
        if v0 is None:
            v0 = float(vol.sum())
        inside = x <= x_end
        vol_in[k] = vol[inside].sum()
        # kernel depth / velocity on analysis cells
        if inside.any():
            ptree = cKDTree(np.c_[x[inside], y[inside]])
            sm = ctree.sparse_distance_matrix(ptree, 2.0 * h_sph, output_type="coo_matrix")
            w = wendland2d(sm.data / h_sph, h_sph) * vol[inside][sm.col]
            d = np.bincount(sm.row, weights=w, minlength=len(idx))
            mu = np.bincount(sm.row, weights=w * vx[inside][sm.col], minlength=len(idx))
            mv = np.bincount(sm.row, weights=w * vy[inside][sm.col], minlength=len(idx))
            ok = d > 1e-3
            depth[k] = d
            u[k, ok] = mu[ok] / d[ok]
            v[k, ok] = mv[ok] / d[ok]
        # section fluxes, outflow and losses from particle identity between snapshots
        cur = {"ids": ids, "x": x, "y": y, "vol": vol}
        if prev is not None:
            order_p = np.argsort(prev["ids"])
            pos = np.searchsorted(prev["ids"][order_p], ids)
            pos = np.clip(pos, 0, len(order_p) - 1)
            match = prev["ids"][order_p][pos] == ids
            jp = order_p[pos[match]]
            x0, y0, x1, y1, vv = prev["x"][jp], prev["y"][jp], x[match], y[match], vol[match]
            for s_i, s in enumerate(stations):
                cval = s["const"]
                (ax, ay), (bx, by) = s["section"]
                if s["axis"] == "x":
                    a0, a1, o0, o1, lo, hi = x0, x1, y0, y1, min(ay, by), max(ay, by)
                else:
                    a0, a1, o0, o1, lo, hi = y0, y1, x0, x1, min(ax, bx), max(ax, bx)
                fwd = (a0 < cval) & (a1 >= cval)
                bwd = (a0 >= cval) & (a1 < cval)
                cr = fwd | bwd
                with np.errstate(invalid="ignore", divide="ignore"):
                    frac = np.where(cr, (cval - a0) / (a1 - a0), 0.0)
                oc = o0 + frac * (o1 - o0)
                within = (oc >= lo) & (oc <= hi)
                net = (vv * (fwd & within)).sum() - (vv * (bwd & within)).sum()
                q[s_i, k] = s["sign"] * net / tout
            gone = ~np.isin(prev["ids"], ids)
            gx = prev["x"][gone]
            excluded_out += prev["vol"][gone][gx >= x_end - 5 * dp].sum()
            excluded_lost += prev["vol"][gone][gx < x_end - 5 * dp].sum()
        # compact particle positions for the 3-D view (every 60 s): uint16 x/y at 0.5 m, z at 0.02 m, speed at 0.2 m/s
        if abs((times[k] / 60.0) - round(times[k] / 60.0)) < 1e-6:
            spd = np.sqrt(vx ** 2 + vy ** 2 + c["vel.z"] ** 2)
            keep_p = inside
            p3d_t.append(times[k])
            p3d_xyz.append(np.c_[np.clip((x[keep_p] - X0P) / 0.5, 0, 65535), np.clip((y[keep_p] - Y0P) / 0.5, 0, 65535),
                                 np.clip((z[keep_p] - Z0P) / 0.02, 0, 65535)].astype(np.uint16))
            p3d_s.append(np.clip(spd[keep_p] / 0.2, 0, 255).astype(np.uint8))
        out_cum[k] = excluded_out + vol[~inside].sum()
        lost_cum[k] = excluded_lost
        prev = cur
        print(f"[sph-extract] {k + 1}/{nt}  t={times[k]:.0f}s  np={len(ids)}  in-domain {vol_in[k] / 1e6:.3f} Mm3  "
              f"out {out_cum[k] / 1e6:.3f}  lost {lost_cum[k] / 1e6:.4f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "frames.npz"), times=times.astype(np.float32), depth=depth,
                        u=u.astype(np.float16), v=v.astype(np.float16))
    counts = np.array([len(a) for a in p3d_xyz], np.int32)
    np.savez_compressed(os.path.join(out_dir, "particles3d.npz"), times=np.array(p3d_t, np.float32), counts=counts,
                        xyz=np.concatenate(p3d_xyz) if p3d_xyz else np.zeros((0, 3), np.uint16),
                        speed=np.concatenate(p3d_s) if p3d_s else np.zeros(0, np.uint8),
                        origin=np.array([X0P, Y0P, Z0P]), scale=np.array([0.5, 0.5, 0.02, 0.2]))
    series = {
        "stations": [{"time": times.tolist(), "discharge": q[i].tolist()} for i in range(len(stations))],
        "outflow": {"time": times.tolist(), "cumulative_m3": out_cum.tolist()},
        "mass_balance": {"initial_volume_m3": v0, "final_volume_m3": float(vol_in[-1]),
                         "outflow_volume_m3": float(out_cum[-1]), "lost_volume_m3": float(lost_cum[-1]),
                         "volume_in_domain_m3": vol_in.tolist(), "lost_cumulative_m3": lost_cum.tolist()},
    }
    with open(os.path.join(out_dir, "series_particles.json"), "w") as f:
        json.dump(series, f)


if __name__ == "__main__":
    main(*sys.argv[1:8])
