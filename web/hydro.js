/* hydrosim viewer core: data decoding, colour ramps, flood map, charts. No external dependencies. */
(function () {
  "use strict";
  const H = (window.Hydro = {});

  // ---------------------------------------------------------------- data
  H.bytes = async function (b64) {
    const bin = atob(b64);
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    const ds = new DecompressionStream("gzip");
    const buf = await new Response(new Blob([u]).stream().pipeThrough(ds)).arrayBuffer();
    return buf;
  };
  H.u8 = async (b64) => new Uint8Array(await H.bytes(b64));
  H.f32 = async (b64) => new Float32Array(await H.bytes(b64));
  H.i32 = async (b64) => new Int32Array(await H.bytes(b64));

  H.fmt = function (v, d = 1) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    return Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  };
  H.mmss = function (sec) {
    const m = Math.floor(sec / 60), s = Math.round(sec % 60);
    const h = Math.floor(m / 60);
    return h ? `T+${h}:${String(m % 60).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `T+${m}:${String(s).padStart(2, "0")}`;
  };
  H.cssVar = (n) => getComputedStyle(document.body).getPropertyValue(n).trim();

  // ---------------------------------------------------------------- colour ramps
  function hex(c) { const n = parseInt(c.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  H.ramp = function (stops, n = 256) {
    const cols = stops.map(hex), out = new Uint8Array(n * 3);
    for (let i = 0; i < n; i++) {
      const t = (i / (n - 1)) * (cols.length - 1), k = Math.min(Math.floor(t), cols.length - 2), f = t - k;
      for (let c = 0; c < 3; c++) out[i * 3 + c] = Math.round(cols[k][c] * (1 - f) + cols[k + 1][c] * f);
    }
    return out;
  };
  H.RAMPS = {
    depth: ["#dff4f7", "#8fd0e3", "#3c96c8", "#1d5aa6", "#1c2f78", "#160f45"],
    speed: ["#fff1b8", "#f7c15a", "#ee7f33", "#cf3f2d", "#8c1c3a", "#3d0b2e"],
    arrival: ["#fde725", "#7ad151", "#22a884", "#2a788e", "#414487", "#440154"],
    diff: ["#1f5aa6", "#8fbfe0", "#f2f2f2", "#f1a27d", "#b4332a"],
  };
  H.HAZARD = [["Low", "#8cc26d"], ["Moderate – danger for some", "#f2d15c"], ["Significant – danger for most", "#ee8a3c"], ["Extreme – danger for all", "#c9302c"]];

  H.rampLegend = function (el, stops, ticks, label) {
    el.innerHTML = "";
    const w = document.createElement("div"); w.className = "ramp";
    const c = document.createElement("canvas"); c.width = 256; c.height = 1;
    const ctx = c.getContext("2d"), lut = H.ramp(stops), img = ctx.createImageData(256, 1);
    for (let i = 0; i < 256; i++) { img.data.set([lut[i * 3], lut[i * 3 + 1], lut[i * 3 + 2], 255], i * 4); }
    ctx.putImageData(img, 0, 0);
    const t = document.createElement("div"); t.className = "ticks";
    t.innerHTML = ticks.map((x) => `<span>${x}</span>`).join("");
    const l = document.createElement("span"); l.textContent = label;
    w.append(c, t); el.append(l, w);
  };
  H.swatchLegend = function (el, items, label) {
    el.innerHTML = `<span>${label}</span><div class="swatches">${items.map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join("")}</div>`;
  };

  // ---------------------------------------------------------------- flood map
  // grid: {nx, ny, dx, x0, y0}; cells: Int32Array of flat indices j*nx+i of active cells
  H.FloodMap = class {
    constructor(host, scen) {
      this.host = host; this.s = scen; const g = scen.grid;
      this.nx = g.nx; this.ny = g.ny;
      host.classList.add("mapbox");
      host.style.aspectRatio = `${g.nx} / ${g.ny}`;
      this.stage = document.createElement("div"); this.stage.className = "mapstage";
      this.terrain = document.createElement("canvas"); this.terrain.className = "terrain";
      this.overlay = document.createElement("canvas"); this.overlay.className = "overlay";
      this.overlay.width = g.nx; this.overlay.height = g.ny;
      this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      this.svg.setAttribute("viewBox", `0 0 ${g.nx} ${g.ny}`); this.svg.setAttribute("preserveAspectRatio", "none");
      this.stage.append(this.terrain, this.overlay, this.svg); host.append(this.stage);
      this.hud = document.createElement("div"); this.hud.className = "map-hud"; host.append(this.hud);
      this.tip = document.createElement("div"); this.tip.className = "tip"; this.tip.hidden = true; host.append(this.tip);
      const z = document.createElement("div"); z.className = "map-zoom";
      z.innerHTML = `<button type="button" aria-label="Zoom in">+</button><button type="button" aria-label="Zoom out">−</button><button type="button" aria-label="Reset view">⌂</button>`;
      host.append(z);
      const [bi, bo, br] = z.querySelectorAll("button");
      bi.onclick = () => this.zoomAt(1.6); bo.onclick = () => this.zoomAt(1 / 1.6); br.onclick = () => this.reset();
      this.ctx = this.overlay.getContext("2d");
      this.img = this.ctx.createImageData(g.nx, g.ny);
      this.px = new Uint32Array(this.img.data.buffer);
      this.k = 1; this.tx = 0; this.ty = 0;
      this.cellPix = new Int32Array(scen.cells.length);
      for (let k = 0; k < scen.cells.length; k++) {
        const c = scen.cells[k], j = Math.floor(c / g.nx), i = c - j * g.nx;
        this.cellPix[k] = (g.ny - 1 - j) * g.nx + i;
      }
      this.pixCell = new Int32Array(g.nx * g.ny).fill(-1);
      for (let k = 0; k < this.cellPix.length; k++) this.pixCell[this.cellPix[k]] = k;
      this._events(); this._resize();
      new ResizeObserver(() => this._resize()).observe(host);
    }
    async setTerrain(pngB64) {
      const im = new Image(); im.src = "data:image/jpeg;base64," + pngB64; await im.decode();
      this.terrain.width = im.width; this.terrain.height = im.height;
      this.terrain.getContext("2d").drawImage(im, 0, 0);
    }
    _resize() {
      const w = this.host.clientWidth; this.base = w / this.nx;
      this.stage.style.width = w + "px"; this.stage.style.height = w * this.ny / this.nx + "px"; this._apply();
    }
    _apply() {
      const W = this.host.clientWidth, Hh = this.host.clientHeight, sw = W * this.k, sh = Hh * this.k;
      this.tx = Math.min(0, Math.max(W - sw, this.tx)); this.ty = Math.min(0, Math.max(Hh - sh, this.ty));
      this.stage.style.transform = `translate(${this.tx}px,${this.ty}px) scale(${this.k})`;
      if (!this.base) return;  // hidden map: nothing to size yet
      this.svg.querySelectorAll("[data-fixed]").forEach((e) => e.setAttribute("stroke-width", (+e.dataset.fixed) / (this.k * this.base)));
      this.svg.querySelectorAll("text").forEach((e) => e.setAttribute("font-size", 12 / (this.k * this.base)));
      this.svg.querySelectorAll("circle[data-r]").forEach((e) => e.setAttribute("r", (+e.dataset.r) / (this.k * this.base)));
    }
    zoomAt(f, cx, cy) {
      const W = this.host.clientWidth, Hh = this.host.clientHeight;
      cx = cx ?? W / 2; cy = cy ?? Hh / 2;
      const k2 = Math.min(40, Math.max(1, this.k * f));
      this.tx = cx - (cx - this.tx) * (k2 / this.k); this.ty = cy - (cy - this.ty) * (k2 / this.k); this.k = k2; this._apply();
    }
    focus(i, j, k) {
      this.k = k; const W = this.host.clientWidth, Hh = this.host.clientHeight;
      this.tx = W / 2 - (i + 0.5) * this.base * k; this.ty = Hh / 2 - (this.ny - 1 - j + 0.5) * this.base * k; this._apply();
    }
    reset() { this.k = 1; this.tx = 0; this.ty = 0; this._apply(); }
    _events() {
      const h = this.host; let drag = null;
      h.addEventListener("wheel", (e) => { e.preventDefault(); const r = h.getBoundingClientRect(); this.zoomAt(e.deltaY < 0 ? 1.25 : 0.8, e.clientX - r.left, e.clientY - r.top); }, { passive: false });
      h.addEventListener("pointerdown", (e) => { if (e.target.closest("button")) return; drag = { x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty }; h.setPointerCapture(e.pointerId); h.classList.add("dragging"); });
      h.addEventListener("pointerup", () => { drag = null; h.classList.remove("dragging"); });
      h.addEventListener("pointerleave", () => { this.tip.hidden = true; });
      h.addEventListener("pointermove", (e) => {
        const r = h.getBoundingClientRect();
        if (drag) { this.tx = drag.tx + e.clientX - drag.x; this.ty = drag.ty + e.clientY - drag.y; this._apply(); return; }
        const sx = (e.clientX - r.left - this.tx) / (this.k * this.base), sy = (e.clientY - r.top - this.ty) / (this.k * this.base);
        const i = Math.floor(sx), row = Math.floor(sy);
        if (i < 0 || row < 0 || i >= this.nx || row >= this.ny || !this.probe) { this.tip.hidden = true; return; }
        const kc = this.pixCell[row * this.nx + i];
        const html = kc >= 0 ? this.probe(kc) : null;
        if (!html) { this.tip.hidden = true; return; }
        this.tip.innerHTML = html; this.tip.hidden = false;
        const x = e.clientX - r.left + 14, y = e.clientY - r.top + 14;
        this.tip.style.left = Math.min(x, r.width - this.tip.offsetWidth - 6) + "px"; this.tip.style.top = Math.min(y, r.height - this.tip.offsetHeight - 6) + "px";
      });
    }
    // values: typed array per active cell; toByte(v) -> 0..255 LUT index or -1 (transparent)
    draw(values, lut, toByte, alpha = 225) {
      const px = this.px; px.fill(0);
      const n = values.length, cp = this.cellPix;
      for (let k = 0; k < n; k++) {
        const b = toByte(values[k]); if (b < 0) continue;
        px[cp[k]] = (alpha << 24) | (lut[b * 3 + 2] << 16) | (lut[b * 3 + 1] << 8) | lut[b * 3];
      }
      this.ctx.putImageData(this.img, 0, 0);
    }
    drawClasses(values, colors, alpha = 215) {
      const rgb = colors.map(hex), px = this.px; px.fill(0);
      for (let k = 0; k < values.length; k++) {
        const c = values[k]; if (c < 0 || c >= rgb.length || Number.isNaN(c)) continue;
        const [r, g, b] = rgb[c | 0]; px[this.cellPix[k]] = (alpha << 24) | (b << 16) | (g << 8) | r;
      }
      this.ctx.putImageData(this.img, 0, 0);
    }
    // geometry overlay in grid units (x right, y down)
    toSvg(x, y) { const g = this.s.grid; return [(x - g.x0) / g.dx, this.ny - (y - g.y0) / g.dx]; }
    addOverlay(scen, opts = {}) {
      const ns = "http://www.w3.org/2000/svg", s = this.svg, mk = (t, a) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); s.append(e); return e; };
      if (scen.roads) for (const line of scen.roads) {
        mk("polyline", { points: line.map((p) => this.toSvg(p[0], p[1]).join(",")).join(" "), fill: "none", stroke: "rgba(60,60,60,.55)", "data-fixed": 0.9 });
      }
      if (scen.buildings) {
        const d = scen.buildings.map((p) => { const [x, y] = this.toSvg(p[0], p[1]); return `M${x.toFixed(1)} ${y.toFixed(1)}h.35`; }).join("");
        mk("path", { d, stroke: "rgba(70,55,50,.38)", "stroke-linecap": "square", "data-fixed": 1.1 });
      }
      const th = scen.thalweg; mk("polyline", { points: th.map((p) => this.toSvg(p[0], p[1]).join(",")).join(" "), fill: "none", stroke: "rgba(255,255,255,.6)", "stroke-dasharray": "4 3", "data-fixed": 0.8 });
      const [dx, dy] = this.toSvg(0, 0);
      mk("circle", { cx: dx, cy: dy, "data-r": 5, fill: "#fff", stroke: "#111", "data-fixed": 1.5 });
      mk("text", { x: dx + 1, y: dy - 6, fill: "#111", "font-weight": 700 }).textContent = "Dam";
      for (const st of scen.stations) {
        const a = this.toSvg(...st.section[0]), b = this.toSvg(...st.section[1]);
        mk("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: "#fff", "data-fixed": 2.4 });
        mk("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: "#111", "data-fixed": 1.2 });
        const t = mk("text", { x: b[0] + 1.5, y: b[1] - 1.5, fill: "#111", "paint-order": "stroke", stroke: "#fff", "stroke-width": 0.35 });
        t.textContent = st.name.replace(/ \(.*\)/, "");
      }
      if (scen.places) for (const p of scen.places) {
        const [x, y] = this.toSvg(p[0], p[1]);
        mk("circle", { cx: x, cy: y, "data-r": 2.4, fill: "#222" });
        const t = mk("text", { x: x + 2, y: y + 1, fill: "#222", "font-style": "italic", "paint-order": "stroke", stroke: "rgba(255,255,255,.85)", "stroke-width": 0.3 }); t.textContent = p[2];
      }
      this._apply();
    }
  };

  // ---------------------------------------------------------------- charts (SVG)
  function niceTicks(lo, hi, n = 5) {
    if (hi <= lo) hi = lo + 1;
    const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const t = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
    return { ticks: t, lo: Math.min(lo, t[0]), hi: Math.max(hi, t[t.length - 1]) };
  }
  // series: [{name, color, x:[], y:[], dash}] ; opts {xLabel, yLabel, yMin, height, markers:[{x,label}]}
  H.lineChart = function (el, series, opts = {}) {
    if (opts.xMax !== undefined) series = series.map((s) => { const k = s.x.findIndex((v) => v > opts.xMax + 1e-9); return k < 0 ? s : { ...s, x: s.x.slice(0, k), y: s.y.slice(0, k) }; });
    const W = 560, Hh = opts.height || 230, m = { l: 58, r: 14, t: 12, b: 38 };
    const xs = series.flatMap((s) => s.x), ys = series.flatMap((s) => s.y).filter((v) => v !== null && Number.isFinite(v));
    if (!xs.length || !ys.length) { el.innerHTML = `<h3>${opts.title || ""}</h3><p class="note">No data.</p>`; return; }
    const X = niceTicks(opts.xMin ?? Math.min(...xs), opts.xMax ?? Math.max(...xs), 6);
    const Y = niceTicks(opts.yMin ?? Math.min(0, Math.min(...ys)), Math.max(...ys), 5);
    const sx = (v) => m.l + ((v - X.lo) / (X.hi - X.lo)) * (W - m.l - m.r);
    const sy = (v) => Hh - m.b - ((v - Y.lo) / (Y.hi - Y.lo)) * (Hh - m.t - m.b);
    let g = `<g class="grid">${Y.ticks.map((t) => `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(t)}" y2="${sy(t)}"/>`).join("")}</g>`;
    g += `<g class="axis"><line x1="${m.l}" x2="${W - m.r}" y1="${Hh - m.b}" y2="${Hh - m.b}"/></g>`;
    g += Y.ticks.map((t) => `<text x="${m.l - 8}" y="${sy(t) + 4}" text-anchor="end">${H.fmt(t, Math.abs(t) < 10 && t % 1 ? 1 : 0)}</text>`).join("");
    g += X.ticks.map((t) => `<text x="${sx(t)}" y="${Hh - m.b + 16}" text-anchor="middle">${H.fmt(t, 0)}</text>`).join("");
    g += `<text x="${(W + m.l) / 2}" y="${Hh - 4}" text-anchor="middle">${opts.xLabel || ""}</text>`;
    g += `<text transform="translate(12 ${(Hh - m.b) / 2}) rotate(-90)" text-anchor="middle">${opts.yLabel || ""}</text>`;
    for (const s of series) {
      let d = "", pen = false;
      for (let i = 0; i < s.x.length; i++) {
        const v = s.y[i]; if (v === null || !Number.isFinite(v)) { pen = false; continue; }
        d += (pen ? "L" : "M") + sx(s.x[i]).toFixed(1) + " " + sy(v).toFixed(1); pen = true;
      }
      if (s.fill) g += `<path d="${d}L${sx(s.x[s.x.length - 1])} ${sy(Y.lo)}L${sx(s.x[0])} ${sy(Y.lo)}Z" fill="${s.color}" opacity=".12"/>`;
      g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 1.8}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} stroke-linejoin="round"/>`;
    }
    (opts.markers || []).forEach((mk) => { g += `<line x1="${sx(mk.x)}" x2="${sx(mk.x)}" y1="${m.t}" y2="${Hh - m.b}" stroke="currentColor" stroke-dasharray="2 3" opacity=".5"/>`; });
    g += `<line class="cursor" x1="0" x2="0" y1="${m.t}" y2="${Hh - m.b}" stroke="var(--accent)" stroke-width="1" opacity="0"/>`;
    const key = series.length > 1 || opts.key ? `<div class="key">${series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div>` : "";
    el.innerHTML = `${opts.title ? `<h3>${opts.title}</h3>` : ""}<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="${opts.title || ""}">${g}</svg>${key}`;
    const svg = el.querySelector("svg"), cur = svg.querySelector(".cursor");
    el._setCursor = (xv) => { if (xv === null) { cur.setAttribute("opacity", 0); return; } cur.setAttribute("x1", sx(xv)); cur.setAttribute("x2", sx(xv)); cur.setAttribute("opacity", 0.8); };
  };

  H.table = function (el, head, rows) {
    el.innerHTML = `<div class="tablewrap"><table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td class="${i ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  };
})();
