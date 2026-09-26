"""
Scenario geometry shared by both solvers.

Everything that defines the physical problem is derived here, once, from the DEM:
  * DEM reprojected to UTM, depressions filled (priority-flood, Barnes et al. 2014)
  * D8 flow directions + accumulation -> main-river thalweg through the dam site
  * local Cartesian frame (origin at dam, +x pointing down-valley) used by both models
  * dam embankment and Froehlich (2008) breach notch burned into the terrain
  * reservoir = DEM volume below the crest level, hydraulically connected upstream of the dam
  * simulation domain = valley corridor (height above main-river thalweg < margin)
  * monitoring stations / cross-sections at fixed chainages

Delft3D-FLOW and DualSPHysics both read terrain.npz, so they see identical topography,
reservoir, breach and domain limits.
"""
import heapq
import json
import math

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform as warp_transform
from scipy import ndimage
from scipy.spatial import cKDTree

G = 9.81
D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ----------------------------------------------------------------------------- frame
class LocalFrame:
    """Rotation about the dam point: local x = down-valley, y = left bank, z = metres a.s.l."""

    def __init__(self, x0, y0, theta):
        self.x0, self.y0, self.theta = float(x0), float(y0), float(theta)
        self.c, self.s = math.cos(theta), math.sin(theta)

    def to_local(self, X, Y):
        dx, dy = np.asarray(X) - self.x0, np.asarray(Y) - self.y0
        return dx * self.c + dy * self.s, -dx * self.s + dy * self.c

    def to_utm(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        return self.x0 + x * self.c - y * self.s, self.y0 + x * self.s + y * self.c

    def as_dict(self):
        return {"x0_utm": self.x0, "y0_utm": self.y0, "theta_rad": self.theta}

    @classmethod
    def from_dict(cls, d):
        return cls(d["x0_utm"], d["y0_utm"], d["theta_rad"])


# ----------------------------------------------------------------------------- hydrology
def priority_flood(z):
    """Depression filling with an epsilon gradient so every cell drains to the DEM edge."""
    ny, nx = z.shape
    valid = np.isfinite(z)
    filled = np.where(valid, z, np.inf).astype(np.float64)
    closed = ~valid
    border = valid & ~ndimage.binary_erosion(valid, border_value=0)
    pq = [(filled[r, c], r, c) for r, c in zip(*np.nonzero(border))]
    heapq.heapify(pq)
    closed[border] = True
    while pq:
        e, r, c = heapq.heappop(pq)
        for dr, dc in D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx and not closed[rr, cc]:
                closed[rr, cc] = True
                if filled[rr, cc] <= e:
                    filled[rr, cc] = e + 1e-3
                heapq.heappush(pq, (filled[rr, cc], rr, cc))
    filled[~valid] = np.nan
    return filled


def d8_receivers(filled, res):
    """Flat index of the steepest-descent neighbour (-1 at outlets)."""
    ny, nx = filled.shape
    zpad = np.pad(filled, 1, constant_values=np.nan)
    best = np.zeros((ny, nx))
    recv = np.full((ny, nx), -1, dtype=np.int64)
    rows, cols = np.mgrid[0:ny, 0:nx]
    for dr, dc in D8:
        zn = zpad[1 + dr:1 + dr + ny, 1 + dc:1 + dc + nx]
        slope = (filled - zn) / (res * (math.sqrt(2.0) if dr and dc else 1.0))
        slope = np.where(np.isfinite(slope), slope, -np.inf)
        better = slope > best
        best = np.where(better, slope, best)
        recv = np.where(better, (rows + dr) * nx + (cols + dc), recv)
    return recv.ravel()


def flow_accumulation(filled, recv):
    order = np.argsort(-np.nan_to_num(filled, nan=-1e9).ravel(), kind="stable")
    acc = np.ones(filled.size, dtype=np.float64)
    recv_l = recv.tolist()
    acc_l = acc.tolist()
    for idx in order.tolist():
        r = recv_l[idx]
        if r >= 0:
            acc_l[r] += acc_l[idx]
    return np.asarray(acc_l).reshape(filled.shape)


def trace_thalweg(recv, acc, start, shape, res, down_len, up_len):
    """Main stem through `start`: downstream along receivers, upstream along max accumulation."""
    ny, nx = shape
    step = lambda a, b: res * (math.sqrt(2.0) if (a // nx != b // nx and a % nx != b % nx) else 1.0)
    down = [start]
    dist = 0.0
    while dist < down_len:
        nxt = recv[down[-1]]
        if nxt < 0:
            break
        dist += step(down[-1], nxt)
        down.append(nxt)
    donors = {}
    for idx in np.nonzero(recv >= 0)[0]:
        donors.setdefault(int(recv[idx]), []).append(int(idx))
    up = [start]
    dist = 0.0
    accf = acc.ravel()
    while dist < up_len:
        cand = donors.get(up[-1], [])
        if not cand:
            break
        nxt = max(cand, key=lambda i: accf[i])
        dist += step(up[-1], nxt)
        up.append(nxt)
    return up[::-1] + down[1:], len(up) - 1


def chainage(xy):
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    return d


def smooth_polyline(xy, window):
    if len(xy) < window:
        return xy
    k = np.ones(window) / window
    pad = window // 2
    out = xy.copy()
    for j in range(2):
        v = np.pad(xy[:, j], pad, mode="edge")
        out[:, j] = np.convolve(v, k, mode="valid")[: len(xy)]
    return out


# ----------------------------------------------------------------------------- breach
def froehlich_2008(volume_m3, breach_height_m, failure_mode="overtopping"):
    """Froehlich (2008), J. Hydraul. Eng. 134(12): average breach width, side slope and formation time."""
    k0 = 1.3 if failure_mode == "overtopping" else 1.0
    b_avg = 0.27 * k0 * volume_m3 ** 0.32 * breach_height_m ** 0.04
    side = 1.0 if failure_mode == "overtopping" else 0.7
    tf = 63.2 * math.sqrt(volume_m3 / (G * breach_height_m ** 2))
    return {"average_width_m": b_avg, "side_slope_h_per_v": side,
            "bottom_width_m": max(b_avg - side * breach_height_m, 5.0),
            "top_width_m": b_avg + side * breach_height_m, "formation_time_s": tf}


# ----------------------------------------------------------------------------- main build
def _utm_dem(src_path, crs, x0, y0, half_w, up, down, res=30.0):
    with rasterio.open(src_path) as src:
        xmin, xmax = x0 - half_w, x0 + half_w
        ymin, ymax = y0 - down, y0 + up
        nx, ny = int((xmax - xmin) / res), int((ymax - ymin) / res)
        tr = from_origin(xmin, ymax, res, res)
        z = np.full((ny, nx), np.nan, np.float32)
        reproject(rasterio.band(src, 1), z, dst_transform=tr, dst_crs=crs,
                  resampling=Resampling.bilinear, dst_nodata=np.nan)
    z[z <= 1.0] = np.nan  # DEM voids / outside the tile are stored as 0
    return z.astype(np.float64), xmin, ymax, res


def build_scenario(sc, log=print):
    cfg, dom, dam = sc.cfg, sc["domain"], sc["dam"]
    crs = sc["crs"]
    (x0,), (y0,) = warp_transform("EPSG:4326", crs, [dam["lon"]], [dam["lat"]])
    L = dom["reach_length_km"] * 1000.0

    log("[terrain] reprojecting DEM to UTM and filling depressions ...")
    z30, gx0, gy1, gres = _utm_dem(sc.input_path("dem"), crs, x0, y0, L + 5000.0, 15000.0, L + 5000.0)
    filled = priority_flood(z30)
    recv = d8_receivers(filled, gres)
    acc = flow_accumulation(filled, recv)
    ny30, nx30 = z30.shape
    r0, c0 = int((gy1 - y0) / gres), int((x0 - gx0) / gres)
    # snap the dam point to the strongest river channel within ~500 m
    k = int(round(500.0 / gres))
    win = acc[max(r0 - k, 0):r0 + k + 1, max(c0 - k, 0):c0 + k + 1]
    dr, dc = np.unravel_index(np.argmax(win), win.shape)
    r0, c0 = max(r0 - k, 0) + dr, max(c0 - k, 0) + dc
    catchment_km2 = acc[r0, c0] * gres * gres / 1e6
    if catchment_km2 < 5.0:
        raise ValueError(f"The chosen point is not on a river (largest channel within 500 m drains only "
                         f"{catchment_km2:.1f} km2). Place the dam on the river channel itself.")
    path, i_dam = trace_thalweg(recv, acc, r0 * nx30 + c0, z30.shape, gres, L + 3000.0, 12000.0)
    pr, pc = np.divmod(np.asarray(path), nx30)
    PX, PY = gx0 + (pc + 0.5) * gres, gy1 - (pr + 0.5) * gres
    PZ = z30[pr, pc]
    zfill = np.nan_to_num(z30, nan=9999.0)

    def crest_length(k):
        """Length of the valley section (perpendicular to the thalweg) lying below bed + H."""
        a, b = max(k - 5, 0), min(k + 5, len(PX) - 1)
        t = np.array([PX[b] - PX[a], PY[b] - PY[a]])
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        s = np.arange(-3000.0, 3000.0, 10.0)
        zl = ndimage.map_coordinates(zfill, [(gy1 - (PY[k] + s * n[1])) / gres - 0.5,
                                            (PX[k] + s * n[0] - gx0) / gres - 0.5], order=1)
        crest = np.min(PZ[max(k - 2, 0):k + 3]) + dam["height_m"]
        i0 = len(s) // 2
        il, ir = i0, i0
        while il > 0 and zl[il - 1] < crest:
            il -= 1
        while ir < len(s) - 1 and zl[ir + 1] < crest:
            ir += 1
        return s[ir] - s[il]

    search = dam.get("siting_search_m", 0.0)
    if search > 0:
        ch0 = chainage(np.c_[PX, PY]) - chainage(np.c_[PX, PY])[i_dam]
        cands = [k for k in range(i_dam, len(PX)) if ch0[k] <= search]
        lens = [crest_length(k) for k in cands]
        k_best = cands[int(np.argmin(lens))]
        log(f"[terrain] dam siting: crest length {lens[0]:.0f} m at the given point, "
            f"{min(lens):.0f} m at chainage {ch0[k_best]:.0f} m -> dam placed there")
        i_dam = k_best
    XD, YD = PX[i_dam], PY[i_dam]
    log(f"[terrain] dam snapped to river cell UTM ({XD:.0f}, {YD:.0f}), catchment {acc[r0, c0] * gres * gres / 1e6:.0f} km2")

    # local frame: +x from the dam toward the thalweg point at chainage L
    ch_utm = chainage(np.c_[PX, PY]) - chainage(np.c_[PX, PY])[i_dam]
    available = float(ch_utm[-1]) - 1000.0
    if available < 3000.0:
        raise ValueError(f"The river can be followed only {max(available, 0) / 1000:.1f} km downstream of this point in the DEM; "
                         "choose a point further upstream or a larger DEM.")
    if L > available:
        log(f"[terrain] the DEM covers only {available / 1000:.1f} km of river downstream; reach shortened from {L / 1000:.0f} km")
        L = available
        dom["reach_length_km"] = L / 1000.0
    i_end = int(np.searchsorted(ch_utm, L))
    theta = math.atan2(PY[i_end] - YD, PX[i_end] - XD)
    fr = LocalFrame(XD, YD, theta)
    tx, ty = fr.to_local(PX, PY)
    txy = smooth_polyline(np.c_[tx, ty], 5)
    tch = chainage(txy)
    tch -= tch[i_dam]
    tz_bed = ndimage.minimum_filter1d(PZ, 3)  # thalweg bed along the path
    x_end = float(np.interp(L, tch, txy[:, 0]))

    # master terrain grid in the local frame
    res = dom["terrain_resolution_m"]
    lat = dom["max_lateral_distance_m"]
    keep = tch <= L + 1.0
    xmin, xmax = txy[keep, 0].min() - lat, x_end
    ymin, ymax = txy[keep, 1].min() - lat, txy[keep, 1].max() + lat
    xs = np.arange(math.floor(xmin / res) * res, xmax + 0.5 * res, res)
    ys = np.arange(math.floor(ymin / res) * res, ymax + res, res)
    Xl, Yl = np.meshgrid(xs, ys)
    Xu, Yu = fr.to_utm(Xl, Yl)
    rowf, colf = (gy1 - Yu) / gres - 0.5, (Xu - gx0) / gres - 0.5
    z_orig = ndimage.map_coordinates(np.nan_to_num(z30, nan=9999.0), [rowf, colf], order=1, mode="nearest")
    log(f"[terrain] local grid {len(xs)} x {len(ys)} @ {res} m, x in [{xs[0]:.0f}, {xs[-1]:.0f}], y in [{ys[0]:.0f}, {ys[-1]:.0f}]")

    # nearest thalweg point for every node
    tree = cKDTree(txy)
    dist, near = tree.query(np.c_[Xl.ravel(), Yl.ravel()])
    dist, near = dist.reshape(Xl.shape), near.reshape(Xl.shape)
    node_ch = tch[near]
    node_zthal = tz_bed[near]

    # ------------------------------------------------------------------ dam + breach
    z_bed = float(np.min(PZ[max(0, i_dam - 2):i_dam + 3]))
    H = float(dam["height_m"])
    z_crest = z_bed + H
    j_a, j_b = np.searchsorted(tch, [-150.0, 150.0])
    tvec = txy[min(j_b, len(txy) - 1)] - txy[j_a]
    tvec /= np.linalg.norm(tvec)
    nvec = np.array([-tvec[1], tvec[0]])
    dxy = np.stack([Xl, Yl], -1)
    d_along = dxy @ tvec  # + downstream
    s_axis = dxy @ nvec   # along dam axis
    wc = dam["crest_width_m"]
    su, sd = dam["upstream_slope_h_per_v"], dam["downstream_slope_h_per_v"]
    over = np.maximum(0.0, np.abs(d_along) - 0.5 * wc)
    z_body = z_crest - over / np.where(d_along < 0, su, sd)
    # lateral extent: valley section along the axis below the crest, plus 30 m keyed into abutments
    s_line = np.arange(-3000.0, 3000.0, res)
    ax_u, ax_v = fr.to_utm(s_line * nvec[0], s_line * nvec[1])
    z_line = ndimage.map_coordinates(np.nan_to_num(z30, nan=9999.0),
                                     [(gy1 - ax_v) / gres - 0.5, (ax_u - gx0) / gres - 0.5], order=1)
    below = z_line < z_crest
    i0 = int(np.argmin(np.abs(s_line)))
    il, ir = i0, i0
    while il > 0 and below[il - 1]:
        il -= 1
    while ir < len(s_line) - 1 and below[ir + 1]:
        ir += 1
    s_left, s_right = s_line[il] - 30.0, s_line[ir] + 30.0
    in_dam = (s_axis >= s_left) & (s_axis <= s_right) & (z_body > z_orig)
    z_mod = np.where(in_dam, z_body, z_orig)

    # reservoir volume (before the notch is cut) -> breach dimensions -> notch
    def reservoir_mask(zfield):
        wet = (zfield < z_crest) & (d_along < 0.0)
        lab, _ = ndimage.label(wet)
        seed = np.unravel_index(np.argmin(np.where(wet & (np.abs(s_axis) < 60) & (d_along > -400), np.hypot(d_along + 150, s_axis), np.inf)), wet.shape)
        return lab == lab[seed]

    res_mask = reservoir_mask(z_mod)
    volume = float(np.sum(z_crest - z_mod[res_mask]) * res * res)
    br = froehlich_2008(volume, H, sc["breach"]["failure_mode"])
    br["side_slope_h_per_v"] = sc["breach"].get("side_slope_h_per_v", br["side_slope_h_per_v"])
    br["bottom_width_m"] = max(br["average_width_m"] - br["side_slope_h_per_v"] * H, 5.0)
    br["top_width_m"] = br["bottom_width_m"] + 2 * br["side_slope_h_per_v"] * H
    z_notch = z_bed + np.maximum(0.0, np.abs(s_axis) - 0.5 * br["bottom_width_m"]) / br["side_slope_h_per_v"]
    notch = in_dam & (np.abs(s_axis) < 0.5 * br["top_width_m"] + res)
    z_mod = np.where(notch, np.maximum(z_orig, np.minimum(z_mod, z_notch)), z_mod)
    res_mask = reservoir_mask(z_mod) & ~(notch & (d_along >= 0))
    volume = float(np.sum(z_crest - z_mod[res_mask]) * res * res)
    log(f"[terrain] bed {z_bed:.1f} m, crest {z_crest:.1f} m, reservoir {volume / 1e6:.2f} Mm3, "
        f"area {res_mask.sum() * res * res / 1e6:.2f} km2")
    log(f"[terrain] Froehlich breach: avg width {br['average_width_m']:.1f} m, bottom {br['bottom_width_m']:.1f} m, "
        f"top {br['top_width_m']:.1f} m (Tf would be {br['formation_time_s'] / 60:.0f} min; run as instantaneous)")

    # ------------------------------------------------------------------ domain
    margin = dom["margin_above_thalweg_m"]
    corridor = (z_orig - node_zthal < margin) & (dist < lat) & (node_ch > 0) & (Xl <= x_end)
    upstream = (z_mod < z_crest + 10.0) & (node_ch <= 0)
    domain = corridor | upstream | res_mask | in_dam
    lab, _ = ndimage.label(domain)
    jd, idam = np.argmin(np.abs(ys)), np.argmin(np.abs(xs))
    domain = lab == lab[jd, idam]
    # crop the grid to the domain bounding box (+2 cells)
    jj, ii = np.nonzero(domain)
    j0, j1 = max(jj.min() - 2, 0), min(jj.max() + 3, len(ys))
    i0_, i1_ = max(ii.min() - 2, 0), len(xs)
    sl = (slice(j0, j1), slice(i0_, i1_))
    xs, ys = xs[i0_:i1_], ys[j0:j1]
    arrays = dict(z_orig=z_orig[sl], z=z_mod[sl], domain=domain[sl], reservoir=res_mask[sl],
                  dam=in_dam[sl], notch=notch[sl], chainage=node_ch[sl], d_along=d_along[sl], s_axis=s_axis[sl])
    log(f"[terrain] domain {arrays['domain'].sum() * res * res / 1e6:.1f} km2 on a {len(xs)} x {len(ys)} grid")

    # ------------------------------------------------------------------ analysis grid
    # Common cell grid: Delft3D computes on it, SPH particles are binned onto it.
    # Cell (j, i) covers terrain nodes (2j..2j+1, 2i..2i+1): corners at xs[0] - res/2 + i * dxa.
    dxa = float(sc["delft3d"]["cell_size_m"])
    f = int(round(dxa / res))
    nyc, nxc = len(ys) // f, len(xs) // f

    def blocks(a):
        return a[: nyc * f, : nxc * f].reshape(nyc, f, nxc, f).swapaxes(1, 2).reshape(nyc, nxc, f * f)

    zb = blocks(arrays["z"])
    cell_dam = blocks(arrays["dam"]).any(-1) & ~blocks(arrays["notch"]).any(-1)
    cell_z = np.where(cell_dam, zb.max(-1), zb.mean(-1))  # keep the crest watertight on the coarse grid
    cell_domain = blocks(arrays["domain"]).any(-1)
    cell_res = blocks(arrays["reservoir"]).any(-1) & (cell_z < z_crest) & (blocks(arrays["d_along"]).max(-1) < 0)
    gx0c, gy0c = float(xs[0] - 0.5 * res), float(ys[0] - 0.5 * res)
    xc = gx0c + (np.arange(nxc) + 0.5) * dxa
    yc = gy0c + (np.arange(nyc) + 0.5) * dxa
    arrays.update(cell_z=cell_z, cell_domain=cell_domain, cell_reservoir=cell_res, cell_dam=cell_dam)

    # ------------------------------------------------------------------ stations (grid-aligned sections)
    def section_at(p, t, zref, name, c):
        axis = "x" if abs(t[0]) >= abs(t[1]) else "y"
        if axis == "x":  # line x = const, flow measured along +x * sign
            i_face = int(round((p[0] - gx0c) / dxa))
            icell = min(max(i_face, 1), nxc - 1) - 1  # cell just upstream of the face
            j0 = int((p[1] - gy0c) // dxa)
            col = cell_domain[:, icell] & cell_domain[:, min(icell + 1, nxc - 1)] & (cell_z[:, icell] < zref + margin)
            lo = hi = j0
            while lo > 0 and col[lo - 1]:
                lo -= 1
            while hi < nyc - 1 and col[hi + 1]:
                hi += 1
            const, rng = gx0c + i_face * dxa, [lo, hi]
        else:
            j_face = int(round((p[1] - gy0c) / dxa))
            jcell = min(max(j_face, 1), nyc - 1) - 1
            i0 = int((p[0] - gx0c) // dxa)
            row = cell_domain[jcell, :] & cell_domain[min(jcell + 1, nyc - 1), :] & (cell_z[jcell, :] < zref + margin)
            lo = hi = i0
            while lo > 0 and row[lo - 1]:
                lo -= 1
            while hi < nxc - 1 and row[hi + 1]:
                hi += 1
            const, rng = gy0c + j_face * dxa, [lo, hi]
        sign = 1 if (t[0] if axis == "x" else t[1]) > 0 else -1
        a = [const, gy0c + rng[0] * dxa] if axis == "x" else [gx0c + rng[0] * dxa, const]
        b = [const, gy0c + (rng[1] + 1) * dxa] if axis == "x" else [gx0c + (rng[1] + 1) * dxa, const]
        return {"name": name, "chainage_m": float(c), "x": float(p[0]), "y": float(p[1]), "bed_m": float(zref),
                "axis": axis, "face_index": i_face if axis == "x" else j_face, "cell_range": [int(rng[0]), int(rng[1])],
                "const": float(const), "sign": sign, "section": [a, b]}

    def tangent_at(c):
        q = np.array([np.interp(c + 150, tch, txy[:, 0]), np.interp(c + 150, tch, txy[:, 1])])
        r = np.array([np.interp(c - 150, tch, txy[:, 0]), np.interp(c - 150, tch, txy[:, 1])])
        return (q - r) / np.linalg.norm(q - r)

    stations = [section_at(np.array([np.interp(c, tch, txy[:, 0]), np.interp(c, tch, txy[:, 1])]),
                           tvec if c < 200 else tangent_at(c), float(np.interp(c, tch, tz_bed)),
                           "Breach outlet" if c < 200 else f"S{k} ({c / 1000:g} km)", c)
                for k, c in enumerate([150.0] + [km * 1000.0 for km in sc["simulation"]["stations_km"] if km * 1000 < L - 500])]

    keep = (tch >= -12000) & (tch <= L)
    np.savez_compressed(sc.terrain_file, xs=xs, ys=ys, xc=xc, yc=yc,
                        thalweg=np.c_[txy[keep], tch[keep], tz_bed[keep]], **arrays)
    meta = {
        "frame": fr.as_dict(), "crs": crs, "resolution_m": res,
        "grid": {"nx": len(xs), "ny": len(ys), "x0": float(xs[0]), "y0": float(ys[0])},
        "analysis_grid": {"nx": nxc, "ny": nyc, "dx": dxa, "x0_corner": gx0c, "y0_corner": gy0c,
                          "active_cells": int(cell_domain.sum()),
                          "reservoir_volume_m3": float(np.sum(z_crest - cell_z[cell_res]) * dxa * dxa)},
        "x_end": x_end, "reach_length_m": L,
        "dam": {"x": 0.0, "y": 0.0, "utm": [XD, YD], "bed_m": z_bed, "crest_m": z_crest, "height_m": H,
                "axis_tangent": tvec.tolist(), "axis_normal": nvec.tolist(),
                "axis_s_range": [float(s_left), float(s_right)], "crest_width_m": wc,
                "upstream_slope": su, "downstream_slope": sd},
        "breach": br | {"invert_m": z_bed, "timing": sc["breach"]["timing"]},
        "reservoir": {"volume_m3": volume, "area_m2": float(res_mask.sum() * res * res), "water_level_m": z_crest},
        "domain_area_m2": float(arrays["domain"].sum() * res * res),
        "stations": stations,
    }
    with open(sc.meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_terrain(sc):
    d = np.load(sc.terrain_file)
    return {k: d[k] for k in d.files}
