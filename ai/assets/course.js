/* AI from Scratch — shared runtime.
 * Provides: error capture (for headless checks), KaTeX rendering, lesson chrome
 * (topbar, prev/next, mark-complete), declarative widgets (.quiz, .numcheck),
 * the Plot canvas helper, sliders, a seeded RNG and tiny linear-algebra helpers.
 * Lesson code: put it in   Course.ready(() => { ... })   at the end of <body>.
 */
(function () {
  "use strict";

  // ---------- error capture (read by tools/check_pages) ----------
  const errors = [];
  function recordError(msg) {
    errors.push(String(msg));
    if (document.body) document.body.setAttribute("data-errors", JSON.stringify(errors));
    console.error(msg);
  }
  window.addEventListener("error", e => recordError(e.message + " @" + (e.filename || "").split("/").pop() + ":" + e.lineno));
  window.addEventListener("unhandledrejection", e => recordError("unhandled rejection: " + e.reason));

  // ---------- progress ----------
  const KEY = "aicourse.done.v1";
  const Progress = {
    all() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } },
    isDone(n) { return !!this.all()[n]; },
    set(n, v) { const d = this.all(); if (v) d[n] = Date.now(); else delete d[n]; localStorage.setItem(KEY, JSON.stringify(d)); },
  };

  // ---------- colors ----------
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim() || name; }
  function color(c) { if (!c) return css("fg"); if (/^(c\d|fg|muted|accent|ok|bad|warn|grid|axis|line|panel|bg)$/.test(c)) return css(c); return c; }
  function withAlpha(c, a) {
    c = color(c);
    if (c.startsWith("#")) {
      let h = c.slice(1); if (h.length === 3) h = h.split("").map(x => x + x).join("");
      const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
      return `rgba(${r},${g},${b},${a})`;
    }
    return c;
  }

  // ---------- Plot ----------
  // World-coordinate 2-D canvas. new Plot(canvasOrSelector, {xmin,xmax,ymin,ymax,width,height,equal})
  class Plot {
    constructor(target, o = {}) {
      this.canvas = typeof target === "string" ? document.querySelector(target) : target;
      if (!this.canvas) throw new Error("Plot: canvas not found: " + target);
      this.W = o.width || +this.canvas.getAttribute("width") || 640;
      this.H = o.height || +this.canvas.getAttribute("height") || 400;
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = this.W * dpr; this.canvas.height = this.H * dpr;
      this.canvas.style.aspectRatio = this.W + " / " + this.H;
      this.ctx = this.canvas.getContext("2d");
      this.ctx.scale(dpr, dpr);
      this.pad = o.pad ?? 0;
      this.setView(o.xmin ?? -5, o.xmax ?? 5, o.ymin ?? -5, o.ymax ?? 5, o.equal ?? false);
      this.handles = []; this.drawFn = null; this._drag = null;
      this._installPointer();
    }
    setView(xmin, xmax, ymin, ymax, equal) {
      if (equal) { // expand the tighter axis so that 1 unit is the same in x and y
        const sx = (this.W - 2 * this.pad) / (xmax - xmin), sy = (this.H - 2 * this.pad) / (ymax - ymin);
        if (sx > sy) { const c = (xmin + xmax) / 2, h = (this.W - 2 * this.pad) / sy / 2; xmin = c - h; xmax = c + h; }
        else { const c = (ymin + ymax) / 2, h = (this.H - 2 * this.pad) / sx / 2; ymin = c - h; ymax = c + h; }
      }
      Object.assign(this, { xmin, xmax, ymin, ymax });
    }
    X(x) { return this.pad + (x - this.xmin) / (this.xmax - this.xmin) * (this.W - 2 * this.pad); }
    Y(y) { return this.H - this.pad - (y - this.ymin) / (this.ymax - this.ymin) * (this.H - 2 * this.pad); }
    toWorld(px, py) {
      return [this.xmin + (px - this.pad) / (this.W - 2 * this.pad) * (this.xmax - this.xmin),
              this.ymin + (this.H - this.pad - py) / (this.H - 2 * this.pad) * (this.ymax - this.ymin)];
    }
    clear() { const c = this.ctx; c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, this.canvas.width, this.canvas.height); c.restore(); return this; }
    _stroke(o, def = "fg") { const c = this.ctx; c.strokeStyle = color(o.color || def); c.lineWidth = o.width || 2; c.setLineDash(o.dash || []); c.globalAlpha = o.alpha ?? 1; }
    grid(o = {}) {
      const c = this.ctx, step = o.step || niceStep(this.xmax - this.xmin), stepY = o.stepY || niceStep(this.ymax - this.ymin);
      c.save(); c.strokeStyle = color("grid"); c.lineWidth = 1;
      for (let x = Math.ceil(this.xmin / step) * step; x <= this.xmax; x += step) { c.beginPath(); c.moveTo(this.X(x), 0); c.lineTo(this.X(x), this.H); c.stroke(); }
      for (let y = Math.ceil(this.ymin / stepY) * stepY; y <= this.ymax; y += stepY) { c.beginPath(); c.moveTo(0, this.Y(y)); c.lineTo(this.W, this.Y(y)); c.stroke(); }
      c.restore(); return this;
    }
    axes(o = {}) {
      const c = this.ctx; c.save(); c.strokeStyle = color("axis"); c.lineWidth = 1.2;
      c.beginPath(); c.moveTo(0, this.Y(0)); c.lineTo(this.W, this.Y(0)); c.moveTo(this.X(0), 0); c.lineTo(this.X(0), this.H); c.stroke();
      if (o.ticks !== false) {
        const sx = o.step || niceStep(this.xmax - this.xmin), sy = o.stepY || niceStep(this.ymax - this.ymin);
        c.fillStyle = color("muted"); c.font = "11px " + css("mono"); c.textAlign = "center"; c.textBaseline = "top";
        const y0 = clamp(this.Y(0) + 3, 2, this.H - 14), x0 = clamp(this.X(0) - 4, 24, this.W - 2);
        for (let x = Math.ceil(this.xmin / sx) * sx; x <= this.xmax; x += sx) if (Math.abs(x) > 1e-9) c.fillText(fmt(x), this.X(x), y0);
        c.textAlign = "right"; c.textBaseline = "middle";
        for (let y = Math.ceil(this.ymin / sy) * sy; y <= this.ymax; y += sy) if (Math.abs(y) > 1e-9) c.fillText(fmt(y), x0, this.Y(y));
      }
      if (o.xlabel) this.label(this.W - 6, this.Y(0) - 8, o.xlabel, { align: "right", px: true });
      if (o.ylabel) this.label(this.X(0) + 6, 12, o.ylabel, { align: "left", px: true });
      c.restore(); return this;
    }
    line(x1, y1, x2, y2, o = {}) { const c = this.ctx; c.save(); this._stroke(o); c.beginPath(); c.moveTo(this.X(x1), this.Y(y1)); c.lineTo(this.X(x2), this.Y(y2)); c.stroke(); c.restore(); return this; }
    // infinite line through (x0,y0) with direction (dx,dy)
    ray(x0, y0, dx, dy, o = {}) { const L = 4 * Math.max(this.xmax - this.xmin, this.ymax - this.ymin) / Math.hypot(dx, dy || 1e-12); return this.line(x0 - L * dx, y0 - L * dy, x0 + L * dx, y0 + L * dy, o); }
    arrow(x0, y0, x1, y1, o = {}) {
      const c = this.ctx, X0 = this.X(x0), Y0 = this.Y(y0), X1 = this.X(x1), Y1 = this.Y(y1);
      const len = Math.hypot(X1 - X0, Y1 - Y0); if (len < 0.5) return this;
      const head = Math.min(o.head || 11, len * 0.45), ang = Math.atan2(Y1 - Y0, X1 - X0);
      c.save(); this._stroke(o, "c1"); c.fillStyle = color(o.color || "c1");
      c.beginPath(); c.moveTo(X0, Y0); c.lineTo(X1 - head * 0.8 * Math.cos(ang), Y1 - head * 0.8 * Math.sin(ang)); c.stroke();
      c.setLineDash([]); c.beginPath(); c.moveTo(X1, Y1);
      c.lineTo(X1 - head * Math.cos(ang - 0.42), Y1 - head * Math.sin(ang - 0.42));
      c.lineTo(X1 - head * Math.cos(ang + 0.42), Y1 - head * Math.sin(ang + 0.42)); c.closePath(); c.fill();
      c.restore();
      if (o.label) this.label(x1, y1, o.label, { color: o.color || "c1", dx: 8 * Math.cos(ang) + 6, dy: 8 * Math.sin(ang) - 6 });
      return this;
    }
    point(x, y, o = {}) {
      const c = this.ctx; c.save(); c.globalAlpha = o.alpha ?? 1; c.fillStyle = color(o.color || "c1");
      c.beginPath(); c.arc(this.X(x), this.Y(y), o.r || 4, 0, 2 * Math.PI); c.fill();
      if (o.stroke) { c.strokeStyle = color(o.stroke); c.lineWidth = 1.5; c.stroke(); }
      c.restore(); if (o.label) this.label(x, y, o.label, { color: o.color || "c1", dx: 8, dy: -8 }); return this;
    }
    label(x, y, s, o = {}) {
      const c = this.ctx; c.save(); c.fillStyle = color(o.color || "fg"); c.font = o.font || "600 13px -apple-system, Helvetica, sans-serif";
      c.textAlign = o.align || "left"; c.textBaseline = o.baseline || "middle";
      const px = o.px ? x : this.X(x), py = o.px ? y : this.Y(y);
      if (o.bg) { const w = c.measureText(s).width; c.fillStyle = withAlpha(o.bg === true ? "bg" : o.bg, .85); const ax = c.textAlign === "center" ? w / 2 : c.textAlign === "right" ? w : 0; c.fillRect(px + (o.dx || 0) - ax - 3, py + (o.dy || 0) - 9, w + 6, 18); c.fillStyle = color(o.color || "fg"); }
      c.fillText(s, px + (o.dx || 0), py + (o.dy || 0)); c.restore(); return this;
    }
    text(x, y, s, o = {}) { return this.label(x, y, s, { font: "13px -apple-system, Helvetica, sans-serif", ...o }); }
    path(pts, o = {}) {
      if (!pts.length) return this; const c = this.ctx; c.save(); this._stroke(o, "c1");
      c.beginPath(); pts.forEach(([x, y], i) => i ? c.lineTo(this.X(x), this.Y(y)) : c.moveTo(this.X(x), this.Y(y)));
      if (o.close) c.closePath();
      if (o.fill) { c.fillStyle = withAlpha(o.fill, o.fillAlpha ?? 0.18); c.fill(); }
      if (o.stroke !== false) c.stroke(); c.restore(); return this;
    }
    polygon(pts, o = {}) { return this.path(pts, { close: true, fill: o.fill || o.color || "c1", ...o }); }
    fn(f, o = {}) {
      const a = o.xmin ?? this.xmin, b = o.xmax ?? this.xmax, n = o.samples || 400, pts = [];
      for (let i = 0; i <= n; i++) { const x = a + (b - a) * i / n, y = f(x); if (isFinite(y)) pts.push([x, y]); else { if (pts.length) this.path(pts, o); pts.length = 0; } }
      return this.path(pts, o);
    }
    circle(x, y, r, o = {}) {
      const c = this.ctx; c.save(); this._stroke(o, "c1");
      c.beginPath(); c.ellipse(this.X(x), this.Y(y), Math.abs(this.X(x + r) - this.X(x)), Math.abs(this.Y(y + r) - this.Y(y)), 0, 0, 2 * Math.PI);
      if (o.fill) { c.fillStyle = withAlpha(o.fill, o.fillAlpha ?? 0.18); c.fill(); } if (o.stroke !== false) c.stroke(); c.restore(); return this;
    }
    rect(x0, y0, x1, y1, o = {}) { return this.polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], o); }
    // Contours / density: f(x,y) -> value, painted as a coloured field.
    field(f, o = {}) {
      const c = this.ctx, res = o.res || 4, vmin = o.vmin, vmax = o.vmax, vals = [];
      let lo = Infinity, hi = -Infinity;
      for (let py = 0; py < this.H; py += res) for (let px = 0; px < this.W; px += res) {
        const [x, y] = this.toWorld(px + res / 2, py + res / 2), v = f(x, y); vals.push(v); if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      }
      lo = vmin ?? lo; hi = vmax ?? hi; let i = 0; c.save();
      for (let py = 0; py < this.H; py += res) for (let px = 0; px < this.W; px += res) {
        const t = (vals[i++] - lo) / (hi - lo || 1); c.fillStyle = cmap(o.cmap || "seq", clamp(t, 0, 1), o.alpha ?? 0.9); c.fillRect(px, py, res, res);
      }
      c.restore(); return this;
    }
    // Contour lines of f at given levels (marching squares, straight segments).
    contour(f, levels, o = {}) {
      const c = this.ctx, res = o.res || 6, nx = Math.ceil(this.W / res) + 1, ny = Math.ceil(this.H / res) + 1, g = [];
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const [x, y] = this.toWorld(i * res, j * res); g.push(f(x, y)); }
      c.save(); this._stroke({ width: 1.2, color: "muted", ...o });
      for (const L of levels) {
        c.beginPath();
        for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
          const v = [g[j * nx + i], g[j * nx + i + 1], g[(j + 1) * nx + i + 1], g[(j + 1) * nx + i]];
          const p = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], cross = [];
          for (let k = 0; k < 4; k++) { const a = v[k] - L, b = v[(k + 1) % 4] - L; if ((a < 0) !== (b < 0)) { const t = a / (a - b); cross.push([(p[k][0] + t * (p[(k + 1) % 4][0] - p[k][0])) * res, (p[k][1] + t * (p[(k + 1) % 4][1] - p[k][1])) * res]); } }
          for (let k = 0; k + 1 < cross.length; k += 2) { c.moveTo(cross[k][0], cross[k][1]); c.lineTo(cross[k + 1][0], cross[k + 1][1]); }
        }
        c.stroke();
      }
      c.restore(); return this;
    }
    // Heatmap of matrix M into world rect [x0,x1]x[y0,y1] (row 0 at top).
    heatmap(M, o = {}) {
      const rows = M.length, cols = M[0].length, x0 = o.x0 ?? this.xmin, x1 = o.x1 ?? this.xmax, y0 = o.y0 ?? this.ymin, y1 = o.y1 ?? this.ymax;
      let lo = o.vmin, hi = o.vmax; if (lo === undefined || hi === undefined) { const f = M.flat(); lo = lo ?? Math.min(...f); hi = hi ?? Math.max(...f); }
      const c = this.ctx, cw = (x1 - x0) / cols, ch = (y1 - y0) / rows; c.save();
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        const t = (M[i][j] - lo) / (hi - lo || 1), X = this.X(x0 + j * cw), Y = this.Y(y1 - i * ch);
        c.fillStyle = cmap(o.cmap || "seq", clamp(t, 0, 1)); c.fillRect(X, Y, this.X(x0 + (j + 1) * cw) - X + 0.5, this.Y(y1 - (i + 1) * ch) - Y + 0.5);
        if (o.values) { c.fillStyle = t > 0.6 && (o.cmap || "seq") === "seq" ? "#fff" : color("fg"); c.font = (o.font || "11px ") + css("mono"); c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(fmt(M[i][j], o.digits ?? 2), this.X(x0 + (j + .5) * cw), this.Y(y1 - (i + .5) * ch)); }
      }
      if (o.rowLabels) o.rowLabels.forEach((s, i) => this.label(x0, y1 - (i + .5) * ch, s, { align: "right", dx: -6, font: "12px " + css("mono") }));
      if (o.colLabels) o.colLabels.forEach((s, j) => this.label(x0 + (j + .5) * cw, y1, s, { align: "center", dy: -10, font: "12px " + css("mono") }));
      c.restore(); return this;
    }
    // Bar chart in world coords: bar i spans [i+gap, i+1-gap].
    bars(vals, o = {}) {
      vals.forEach((v, i) => {
        const col = Array.isArray(o.color) ? o.color[i] : (o.color || "c1");
        this.rect(i + (o.gap ?? .15), 0, i + 1 - (o.gap ?? .15), v, { fill: col, fillAlpha: o.fillAlpha ?? .75, color: col, width: 1 });
        if (o.labels) this.label(i + .5, this.ymin, o.labels[i], { align: "center", dy: -9, font: "12px " + css("mono"), color: "muted" });
        if (o.values) this.label(i + .5, v, fmt(v, o.digits ?? 2), { align: "center", dy: v >= 0 ? -9 : 9, font: "11px " + css("mono") });
      });
      return this;
    }
    // Draggable handles: handles = [{x,y,color,label,constrain?:(x,y)=>[x,y]}]. Calls this.render() on drag.
    draggable(handles, onChange) { this.handles = handles; this.onDrag = onChange; return this; }
    drawHandles() { for (const h of this.handles) { if (h.hidden) continue; this.point(h.x, h.y, { r: h.r || 7, color: h.color || "c2", stroke: "bg" }); } return this; }
    render(fn) { if (fn) this.drawFn = fn; if (this.drawFn) { this.clear(); this.drawFn(this); } return this; }
    _installPointer() {
      const cv = this.canvas, pos = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * this.W / r.width, (e.clientY - r.top) * this.H / r.height]; };
      cv.addEventListener("pointerdown", e => {
        const [px, py] = pos(e); let best = null, bd = 18;
        for (const h of this.handles) { const d = Math.hypot(this.X(h.x) - px, this.Y(h.y) - py); if (d < bd && !h.fixed) { bd = d; best = h; } }
        if (best) { this._drag = best; cv.setPointerCapture(e.pointerId); e.preventDefault(); }
        else if (this.onClick) { const [x, y] = this.toWorld(px, py); this.onClick(x, y, e); }
      });
      cv.addEventListener("pointermove", e => {
        const [px, py] = pos(e);
        if (!this._drag) { cv.style.cursor = this.handles.some(h => !h.fixed && Math.hypot(this.X(h.x) - px, this.Y(h.y) - py) < 18) ? "grab" : (this.onClick ? "crosshair" : "default"); return; }
        let [x, y] = this.toWorld(px, py); if (this._drag.snap) { x = Math.round(x / this._drag.snap) * this._drag.snap; y = Math.round(y / this._drag.snap) * this._drag.snap; }
        if (this._drag.constrain) [x, y] = this._drag.constrain(x, y);
        this._drag.x = x; this._drag.y = y; if (this.onDrag) this.onDrag(this._drag); this.render();
      });
      const up = () => { this._drag = null; }; cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
    }
  }
  function niceStep(span) { const raw = span / 10, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p; return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p; }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function fmt(v, d) { if (d === undefined) { return Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : (+v.toPrecision(3)).toString(); } return (+v).toFixed(d); }
  // colormaps: seq (light->accent), div (c4 <- white -> c1), viridis-ish
  function cmap(name, t, a = 1) {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const lerp = (A, B, t) => A.map((v, i) => Math.round(v + (B[i] - v) * t));
    let rgb;
    if (name === "div") { const neg = [214, 51, 108], mid = dark ? [30, 32, 37] : [250, 250, 248], pos = [47, 111, 222]; rgb = t < .5 ? lerp(neg, mid, t * 2) : lerp(mid, pos, (t - .5) * 2); }
    else if (name === "viridis") { const s = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]; const k = Math.min(3, Math.floor(t * 4)); rgb = lerp(s[k], s[k + 1], t * 4 - k); }
    else { const lo = dark ? [29, 32, 37] : [246, 248, 252], hi = dark ? [110, 160, 255] : [23, 72, 170]; rgb = lerp(lo, hi, t); }
    return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
  }

  // ---------- controls ----------
  // Course.slider(container, {label, min, max, step, value, digits}, onInput) -> input
  function slider(container, o, onInput) {
    const host = typeof container === "string" ? document.querySelector(container) : container;
    const lab = document.createElement("label"), inp = document.createElement("input"), val = document.createElement("span");
    inp.type = "range"; inp.min = o.min ?? 0; inp.max = o.max ?? 1; inp.step = o.step ?? 0.01; inp.value = o.value ?? inp.min;
    val.className = "val"; const show = () => { val.textContent = (+inp.value).toFixed(o.digits ?? (String(inp.step).split(".")[1] || "").length); };
    lab.append(document.createTextNode(o.label || ""), inp, val); host.appendChild(lab); show();
    inp.addEventListener("input", () => { show(); onInput && onInput(+inp.value); });
    inp.set = v => { inp.value = v; show(); }; return inp;
  }
  function button(container, label, onClick, cls = "btn") {
    const host = typeof container === "string" ? document.querySelector(container) : container;
    const b = document.createElement("button"); b.className = cls; b.textContent = label; b.addEventListener("click", onClick); host.appendChild(b); return b;
  }
  function select(container, label, options, onChange) {
    const host = typeof container === "string" ? document.querySelector(container) : container;
    const lab = document.createElement("label"), s = document.createElement("select");
    options.forEach(([v, t]) => { const op = document.createElement("option"); op.value = v; op.textContent = t; s.appendChild(op); });
    lab.append(document.createTextNode(label), s); host.appendChild(lab); s.addEventListener("change", () => onChange(s.value)); return s;
  }
  // Simple animation loop; returns {start, stop, running}
  function animator(step) {
    let id = null; const a = {
      running: false,
      start() { if (a.running) return; a.running = true; const tick = () => { if (!a.running) return; try { if (step() === false) { a.stop(); return; } } catch (e) { a.stop(); recordError(e.stack || e); return; } id = requestAnimationFrame(tick); }; id = requestAnimationFrame(tick); },
      stop() { a.running = false; if (id) cancelAnimationFrame(id); },
    }; return a;
  }

  // ---------- RNG & math ----------
  function rng(seed = 42) {
    let s = seed >>> 0; const u = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    return { uniform: (a = 0, b = 1) => a + (b - a) * u(), normal: (m = 0, sd = 1) => m + sd * Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u()), int: n => Math.floor(u() * n), choice: arr => arr[Math.floor(u() * arr.length)] };
  }
  const LA = {
    dot: (a, b) => a.reduce((s, v, i) => s + v * b[i], 0),
    norm: a => Math.sqrt(a.reduce((s, v) => s + v * v, 0)),
    add: (a, b) => a.map((v, i) => v + b[i]), sub: (a, b) => a.map((v, i) => v - b[i]), scale: (a, k) => a.map(v => v * k),
    matvec: (M, v) => M.map(r => r.reduce((s, x, j) => s + x * v[j], 0)),
    matmul: (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, x, k) => s + x * B[k][j], 0))),
    T: A => A[0].map((_, j) => A.map(r => r[j])),
    det2: M => M[0][0] * M[1][1] - M[0][1] * M[1][0],
    softmax: (v, temp = 1) => { const m = Math.max(...v), e = v.map(x => Math.exp((x - m) / temp)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); },
    // eigen-decomposition of a symmetric 2x2 [[a,b],[b,d]] -> {vals:[l1,l2], vecs:[[x,y],[x,y]]} (l1 >= l2)
    eigSym2: ([[a, b], [, d]]) => { const tr = a + d, disc = Math.sqrt(((a - d) / 2) ** 2 + b * b), l1 = tr / 2 + disc, l2 = tr / 2 - disc; const v = Math.abs(b) > 1e-12 ? [l1 - d, b] : (a >= d ? [1, 0] : [0, 1]); const n = Math.hypot(...v); const v1 = [v[0] / n, v[1] / n]; return { vals: [l1, l2], vecs: [v1, [-v1[1], v1[0]]] }; },
  };

  // ---------- declarative widgets ----------
  function initQuizzes(root = document) {
    root.querySelectorAll(".quiz").forEach(q => {
      const ans = (q.dataset.answer || "").split(",");
      q.querySelectorAll(".opts button").forEach(b => b.addEventListener("click", () => {
        const right = ans.includes(b.dataset.k); b.classList.add(right ? "right" : "wrong"); q.classList.add("answered");
        if (right) q.querySelectorAll(".opts button").forEach(o => { if (ans.includes(o.dataset.k)) o.classList.add("right"); });
      }));
    });
  }
  function initNumchecks(root = document) {
    root.querySelectorAll(".numcheck").forEach(n => {
      if (n.querySelector("input")) return;
      const ev = s => Function('"use strict";return (' + s.replace(/\^/g, "**") + ")")();
      const want = n.dataset.answer.split(",").map(s => ev(s.trim())), tol = +(n.dataset.tol || 1e-3);
      const inp = document.createElement("input"), b = document.createElement("button"), fb = document.createElement("span");
      inp.placeholder = want.length > 1 ? "e.g. 1, 2" : "your answer"; b.className = "btn"; b.textContent = "Check"; fb.className = "fb";
      const check = () => {
        let got; try { got = inp.value.split(/[,\s]+/).filter(Boolean).map(ev); } catch { got = [NaN]; }
        const ok = got.length === want.length && got.every((g, i) => Math.abs(g - want[i]) <= tol * Math.max(1, Math.abs(want[i])));
        fb.textContent = ok ? "✓ correct" : "✗ not quite — try again"; fb.className = "fb " + (ok ? "ok" : "bad");
      };
      b.addEventListener("click", check); inp.addEventListener("keydown", e => { if (e.key === "Enter") check(); });
      n.append(inp, b, fb);
    });
  }

  // ---------- lesson chrome ----------
  function lessonChrome() {
    const n = document.body.dataset.lesson; if (!n || !window.COURSE) return;
    const all = COURSE.all, i = all.findIndex(l => l.n === n), L = all[i]; if (!L) return;
    const file = l => `${l.n}-${l.slug}.html`;
    document.title = `${L.n} · ${L.title} — AI from Scratch`;
    const main = document.querySelector("main");
    const top = document.createElement("div"); top.className = "topbar";
    top.innerHTML = `<a href="../index.html">← AI from Scratch</a><span>${L.partTitle} · Lesson ${L.n} of ${all[all.length - 1].n} · ~${L.hours} h</span>`;
    main.prepend(top);
    // auto TOC
    const toc = document.querySelector("nav.toc, ul.toc");
    if (toc && !toc.children.length) document.querySelectorAll("main h2[id]").forEach(h => { const li = document.createElement("li"); li.innerHTML = `<a href="#${h.id}">${h.textContent}</a>`; toc.appendChild(li); });
    const foot = document.createElement("div"); foot.className = "lesson-footer";
    const prev = all[i - 1], next = all[i + 1];
    foot.innerHTML = `<span>${prev ? `<a class="btn" href="${file(prev)}">← ${prev.n} ${prev.title}</a>` : ""}</span><button class="btn" id="mark-done"></button><span>${next ? `<a class="btn primary" href="${file(next)}">${next.n} ${next.title} →</a>` : ""}</span>`;
    main.appendChild(foot);
    const md = foot.querySelector("#mark-done"), upd = () => { const d = Progress.isDone(n); md.textContent = d ? "✓ Completed" : "Mark lesson complete"; md.className = "btn" + (d ? " done" : ""); };
    md.addEventListener("click", () => { Progress.set(n, !Progress.isDone(n)); upd(); }); upd();
  }

  function renderMath() {
    if (!window.renderMathInElement) { recordError("KaTeX auto-render not loaded"); return; }
    renderMathInElement(document.body, {
      delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }, { left: "$", right: "$", display: false }],
      throwOnError: false, ignoredClasses: ["nomath"],
      errorCallback: (msg, err) => recordError("KaTeX: " + msg + " " + (err && err.message)),
    });
    document.querySelectorAll(".katex-error").forEach(e => recordError("KaTeX error: " + e.getAttribute("title")));
  }

  const readyFns = []; let isReady = false;
  function ready(fn) { if (isReady) run(fn); else readyFns.push(fn); }
  function run(fn) { try { fn(); } catch (e) { recordError(e.stack || e.message || e); } }
  document.addEventListener("DOMContentLoaded", () => {
    run(lessonChrome); run(renderMath); run(initQuizzes); run(initNumchecks);
    isReady = true; readyFns.splice(0).forEach(run);
    document.body.setAttribute("data-errors", JSON.stringify(errors));
    document.body.setAttribute("data-ready", "1");
  });

  window.Course = { ready, Plot, slider, button, select, animator, rng, LA, color, withAlpha, cmap, fmt, clamp, Progress, recordError, renderMath, initQuizzes, initNumchecks };
  window.Plot = Plot; window.LA = LA;
})();
