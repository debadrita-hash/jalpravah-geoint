"""
JalPravah Online: makes the Vercel site (https://jalpravah-geoint.vercel.app) use this workstation, so New scenario,
Run / Delete and Satellite flood mapping work there as they do on http://localhost:8765/.

    python jalpravah_online.py            (or double-click "JalPravah Online.bat")

1. starts the JalPravah server if it is not running
2. opens a Cloudflare quick tunnel to it (cloudflared, downloaded to D:\\delft3d\\tools on first use)
3. stores the tunnel URL, the shared key and the password in the Vercel project and redeploys the site
Keep this window open; closing it takes the workstation offline (the site then shows the saved results).
The server answers tunnelled requests only when they carry the shared key, which only the Vercel site has.
Needs the Vercel CLI signed in once (npx vercel login).
"""
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import jalpravah

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
CLOUDFLARED = TOOLS / "cloudflared.exe"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
LOCAL = "http://127.0.0.1:8765"  # the server listens on IPv4 only
SITE = "https://jalpravah-geoint.vercel.app"


def log(msg):
    print(f"[online] {msg}", flush=True)


def http(url, headers=None, data=None, method=None, timeout=20):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def remote_settings():
    s = json.loads(jalpravah.REMOTE_FILE.read_text()) if jalpravah.REMOTE_FILE.exists() else {}
    if not s.get("key"):
        s["key"] = secrets.token_hex(24)
    if not s.get("password"):
        s["password"] = secrets.token_urlsafe(9)
    jalpravah.REMOTE_FILE.write_text(json.dumps(s, indent=2))
    return s


def ensure_server():
    try:
        http(f"{LOCAL}/api/jobs", timeout=10)
        log("JalPravah server is running")
        return
    except Exception:
        pass
    log("starting the JalPravah server")
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0
    subprocess.Popen([sys.executable, "jalpravah.py", "--no-browser"], cwd=HERE, creationflags=flags,
                     stdout=open(HERE / "jalpravah_server.log", "a"), stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(1)
        try:
            http(f"{LOCAL}/api/jobs", timeout=5)
            return
        except Exception:
            pass
    raise SystemExit("The JalPravah server did not start; see jalpravah_server.log")


def ensure_cloudflared():
    if not CLOUDFLARED.exists():
        TOOLS.mkdir(parents=True, exist_ok=True)
        log("downloading cloudflared (Cloudflare tunnel client)")
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED)
    return CLOUDFLARED


def open_tunnel():
    p = subprocess.Popen([str(ensure_cloudflared()), "tunnel", "--no-autoupdate", "--url", LOCAL],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore")
    t0 = time.time()
    for line in p.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            threading.Thread(target=lambda: [None for _ in p.stdout], daemon=True).start()  # keep draining its log
            return p, m.group(0)
        if time.time() - t0 > 90:
            break
    p.kill()
    raise SystemExit("cloudflared did not report a tunnel URL")


def wait_reachable(url, key):
    for _ in range(60):
        try:
            if http(f"{url}/api/jobs", {"X-JalPravah-Key": key}, timeout=10)[0] == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit(f"The tunnel {url} did not become reachable")


def vercel_auth():
    base = Path(os.environ.get("APPDATA", "")) / "com.vercel.cli" / "Data" / "auth.json"
    token = json.loads(base.read_text())["token"]
    proj = json.loads((HERE / ".vercel" / "project.json").read_text())
    return token, proj["projectId"], proj.get("orgId")


def publish(tunnel, s):
    token, pid, team = vercel_auth()
    q = f"?upsert=true&teamId={team}" if team and team.startswith("team_") else "?upsert=true"
    for k, v in (("JALPRAVAH_BACKEND", tunnel), ("JALPRAVAH_KEY", s["key"]), ("JALPRAVAH_PASSWORD", s["password"])):
        body = json.dumps({"key": k, "value": v, "type": "encrypted", "target": ["production"]}).encode()
        http(f"https://api.vercel.com/v10/projects/{pid}/env{q}", {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, body, "POST")
    log("redeploying the Vercel site with the new tunnel (about a minute)")
    env = dict(os.environ, VERCEL_TELEMETRY_DISABLED="1")
    r = subprocess.run("npx --yes vercel deploy --prod --yes", cwd=HERE, shell=True, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise SystemExit("Vercel deploy failed:\n" + (r.stderr or r.stdout)[-2000:])


def main():
    s = remote_settings()
    ensure_server()
    while True:
        proc, tunnel = open_tunnel()
        log(f"tunnel {tunnel}")
        wait_reachable(tunnel, s["key"])
        publish(tunnel, s)
        log(f"ONLINE: {SITE}  (password for actions: {s['password']})")
        log("keep this window open; Ctrl+C takes the workstation offline")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            log("offline")
            return
        log("the tunnel closed; reopening")


if __name__ == "__main__":
    main()
