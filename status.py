"""
Progress and ETA of every hydrosim run, read from the solvers' own logs.

    python status.py            one-off report
    python status.py --watch    refresh every 60 s (Ctrl+C to stop)
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from hydrosim.config import Scenario

HERE = Path(__file__).resolve().parent


def sh(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30, env=dict(os.environ, MSYS_NO_PATHCONV="1"))
        return r.stdout
    except Exception:
        return ""


def running(name, volume):
    return bool(sh(["docker", "ps", "-q", "-f", f"name={name}", "-f", f"volume={volume}"]).strip())


def fmt_td(sec):
    sec = max(0, int(sec))
    return f"{sec // 3600}h {(sec % 3600) // 60:02d}m"


def delft3d_status(sc):
    planned = float(sc["simulation"]["duration_s"])
    log = sc.delft3d_dir / "case" / "run.log"
    live = running("hydrosim_d3d_solver", f"hydrosim_{sc.name}_d3d")
    tail = log.read_text(errors="ignore")[-3000:] if log.exists() else ""
    if live:
        pct = re.findall(r"([\d.]+)% completed", tail)
        left = re.findall(r"Time to finish\s+(?:(\d+)h)?\s*(\d+)m", tail)
        p = float(pct[-1]) if pct else 0.0
        eta = None
        if left:
            h, m = left[-1]
            eta = dt.datetime.now() + dt.timedelta(hours=int(h or 0), minutes=int(m))
        return "running", planned * p / 100.0, planned, eta, ""
    metrics = sc.results_dir / "delft3d" / "metrics.json"
    finished_ok = "shutting down normally" in tail and "exited abnormally" not in tail
    if finished_ok and metrics.exists() and metrics.stat().st_mtime >= log.stat().st_mtime:
        return "complete", planned, planned, None, ""
    if finished_ok:
        an = sc.delft3d_dir / "analyze.out"
        step = ""
        if an.exists():
            lines = an.read_text(errors="ignore").strip().splitlines()
            step = lines[-1][:80] if lines else ""
        return "analysing", planned, planned, None, "solver finished; post-processing: " + step
    if "exited abnormally" in tail:
        return "FAILED", 0.0, planned, None, "see " + str(log)
    return "not run", 0.0, planned, None, ""


def sph_status(sc):
    planned = float(sc["sph"]["duration_s"])
    vol = f"hydrosim_{sc.name}_sph"
    live = running("hydrosim_sph_solver", vol)
    if vol not in sh(["docker", "volume", "ls", "-q"]).split():
        return "not scheduled", 0.0, planned, None
    out = sh(["docker", "run", "--rm", "-v", f"{vol}:/work:ro", "alpine", "sh", "-c",
              "d=$(cd /work/case && ls -d out out_ext* 2>/dev/null | sort -V | tail -1); grep -E '^[0-9]{5} ' /work/case/$d/Run.out | tail -1"])
    m = re.match(r"\d+\s+([\d.]+)\s+.*\s(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})\s*$", out.strip())
    sim, eta = 0.0, None
    if m:
        sim = float(m.group(1))
        eta = dt.datetime.strptime(m.group(2), "%d-%m-%Y %H:%M:%S").replace(tzinfo=dt.timezone.utc).astimezone().replace(tzinfo=None)
    ext = sc.sph_dir / "extend.log"
    note = ""
    if ext.exists():
        last = ext.read_text(errors="ignore").strip().splitlines()[-1:]
        note = last[0] if last else ""
    if not m and not live:
        return "not run", 0.0, planned, None
    if live:
        tgt = re.findall(r"to (\d+) s in out_ext", note)
        if tgt:
            note = f"extension chunk up to {int(tgt[-1]) / 60:.0f} min (ETA is for this chunk); stops when the SPH flood extent stops growing"
        return "running", sim, planned, eta, note
    if "stopping" in note or "cap" in note:
        return "complete", sim, sim, None, note
    return "stopped", sim, planned, None, note


def report():
    now = dt.datetime.now()
    print(f"\nhydrosim status  {now:%d %b %Y %H:%M}")
    print("-" * 96)
    print(f"{'scenario':32s} {'model':8s} {'state':12s} {'simulated':>18s} {'progress':>9s}  {'ETA (this stage)':>18s}")
    for cfg in sorted((HERE / "config").glob("*.json")):
        sc = Scenario(cfg)
        for model, fn in (("Delft3D", delft3d_status), ("SPH", sph_status)):
            st = fn(sc)
            state, sim, planned, eta = st[:4]
            pct = 100.0 * sim / planned if planned else 0.0
            eta_s = f"{eta:%d %b %H:%M}" if eta else ("-" if state != "running" else "estimating")
            print(f"{sc.name:32s} {model:8s} {state:12s} {sim / 60:7.1f} / {planned / 60:6.1f} min {pct:8.1f}%  {eta_s:>18s}")
            if len(st) > 4 and st[4]:
                print(f"{'':32s} {'':8s} {st[4][:95]}")
    print("-" * 96)
    print("SPH on the 30 km scenario continues automatically after 90 min until its own flood extent stops growing (cap 6 h).")


if __name__ == "__main__":
    if "--watch" in sys.argv:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            report()
            time.sleep(60)
    else:
        report()
