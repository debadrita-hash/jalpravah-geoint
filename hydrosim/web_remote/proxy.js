// JalPravah on Vercel: forwards live requests to the JalPravah workstation through its Cloudflare tunnel.
//
// Environment (set by jalpravah_online.py):
//   JALPRAVAH_BACKEND   tunnel URL of the workstation, e.g. https://xyz.trycloudflare.com (empty: offline)
//   JALPRAVAH_KEY       shared secret; the workstation only answers tunnelled requests that carry it
//   JALPRAVAH_PASSWORD  password that unlocks actions (build, run, delete, satellite mapping) in the browser
//
// Reading is public; anything that changes the workstation needs the password cookie. When the workstation is
// offline, reads fall back to the snapshot of finished results published with the site.
const crypto = require("crypto");

const BACKEND = (process.env.JALPRAVAH_BACKEND || "").replace(/\/+$/, "");
const KEY = process.env.JALPRAVAH_KEY || "";
const PASSWORD = process.env.JALPRAVAH_PASSWORD || "";
const MAX_INLINE = 4000000; // Vercel responses are limited to 4.5 MB; larger files are fetched from the tunnel directly
const OFFLINE = "The JalPravah workstation is offline. Building scenarios, running simulations and satellite mapping resume when JalPravah Online is running on it; the saved results stay available here.";

const session = () => crypto.createHash("sha256").update(`jalpravah:${PASSWORD}:${KEY}`).digest("hex");
const dlToken = (path) => crypto.createHmac("sha256", KEY).update(path).digest("hex").slice(0, 32);

function cookies(req) {
  return Object.fromEntries((req.headers.cookie || "").split(";").map((c) => c.trim().split("=")).filter((c) => c[0]).map(([k, ...v]) => [k, v.join("=")]));
}
const signedIn = (req) => !!PASSWORD && cookies(req).jp_session === session();

function json(res, code, obj) {
  res.statusCode = code;
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(obj));
}

async function snapshot(req, file) {
  const r = await fetch(`https://${req.headers.host}/snapshot/${file}`);
  return r.ok ? r.json() : null;
}

async function offline(req, res, path) {
  let m;
  if (path === "/api/scenarios") return json(res, 200, (await snapshot(req, "scenarios.json")) || []);
  if (path === "/api/jobs" || path === "/api/gee/runs") return json(res, 200, []);
  if (/^\/api\/jobs\/[^/]+\/log$/.test(path)) { res.setHeader("Content-Type", "text/plain; charset=utf-8"); return res.end(""); }
  if (path === "/api/gee/status") return json(res, 200, { ready: false, step: "offline", message: OFFLINE });
  if ((m = path.match(/^\/run\/([^/]+)\/api\/(status|manifest)$/))) {
    const s = await snapshot(req, `run/${m[1]}/${m[2]}.json`);
    if (s) return json(res, 200, s);
  }
  if (path.startsWith("/api/") || path.includes("/api/")) return json(res, 503, { error: OFFLINE });
  res.statusCode = 503;
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.end(`<!doctype html><meta charset="utf-8"><title>JalPravah</title><link rel="stylesheet" href="/hydro.css">
<div class="wrap"><h1>Not available right now</h1><p class="lede">${OFFLINE}</p><p><a href="/">Open the finished scenarios</a></p></div>`);
}

module.exports = async (req, res) => {
  const url = new URL(req.url, "https://x");
  const path = url.searchParams.get("__p") || url.pathname;
  url.searchParams.delete("__p");
  const query = url.searchParams.toString();

  // sign-in state for the page banner
  if (path === "/api/remote") return json(res, 200, { online: !!BACKEND && (await ping()), signedIn: signedIn(req), passwordSet: !!PASSWORD });
  if (path === "/api/remote/login" && req.method === "POST") {
    const pw = (req.body && req.body.password) || "";
    const ok = PASSWORD && pw.length === PASSWORD.length && crypto.timingSafeEqual(Buffer.from(pw), Buffer.from(PASSWORD));
    if (!ok) return json(res, 401, { error: "Wrong password." });
    res.setHeader("Set-Cookie", `jp_session=${session()}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000`);
    return json(res, 200, { ok: true });
  }
  if (path === "/api/remote/logout") {
    res.setHeader("Set-Cookie", "jp_session=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0");
    return json(res, 200, { ok: true });
  }

  if (req.method !== "GET" && req.method !== "HEAD" && !signedIn(req))
    return json(res, 401, { error: "Sign in with the JalPravah password (bar at the top of the page) to build, run or delete." });
  if (!BACKEND || !KEY) return offline(req, res, path);

  const target = `${BACKEND}${path}${query ? "?" + query : ""}`;
  const init = { method: req.method, headers: { "X-JalPravah-Key": KEY }, redirect: "manual", signal: AbortSignal.timeout(55000) };
  if (req.method === "POST") {
    init.headers["Content-Type"] = "application/json";
    init.body = typeof req.body === "string" ? req.body : JSON.stringify(req.body || {});
  }
  let r;
  try {
    r = await fetch(target, init);
  } catch (e) {
    return offline(req, res, path);
  }
  if (r.status === 530 || r.status === 502 || r.status === 1033) return offline(req, res, path); // tunnel gone
  const len = +(r.headers.get("content-length") || 0);
  if (req.method === "GET" && len > MAX_INLINE) {
    r.body && r.body.cancel();
    res.statusCode = 302;
    res.setHeader("Location", `${target}${query ? "&" : "?"}jpdl=${dlToken(path)}`);
    return res.end();
  }
  res.statusCode = r.status;
  for (const h of ["content-type", "content-disposition", "cache-control"]) {
    const v = r.headers.get(h);
    if (v) res.setHeader(h, v);
  }
  res.end(Buffer.from(await r.arrayBuffer()));
};

async function ping() {
  try {
    const r = await fetch(`${BACKEND}/api/jobs`, { headers: { "X-JalPravah-Key": KEY }, signal: AbortSignal.timeout(8000) });
    return r.ok;
  } catch (e) {
    return false;
  }
}
