/* BBOB landscape renderer, shared by the Atlas (index.html) and the Optimizer Race (race.html).

   window.BBOBLandscape = { decode, field, ramp, COOLWARM, LUT, HEIGHT, LEVELS, num, neg, sup, coord, create }

   create(opts) builds the stage inside opts.container: a three.js surface (or a 2-D heat map
   when WebGL is missing), the domain frame, the axis labels, the optimum pin and, when
   opts.overlay is given, a 2-D canvas above the surface on which the page draws its own marks.
   The page owns the animation loop and calls stage.draw() once per frame.
   Needs three.js r128 as window.THREE. */
(function (root) {
"use strict";

const HEIGHT = 0.8;                 // elevation of the highest point, in half-widths of the square
const LEVELS = 65534;               // quantisation codes 1..65535 (see build.py)
const TILT_ELEV = 36 * Math.PI / 180, TILT_TOP = 89.6 * Math.PI / 180;

/* ---------- data ---------- */
function decode(b64) {
  const s = atob(b64), u = new Uint16Array(s.length >> 1);
  for (let i = 0, k = 0; i < u.length; i++, k += 2) u[i] = s.charCodeAt(k) | (s.charCodeAt(k + 1) << 8);
  return u;
}

/* Heights in [0, 1] for one function and scale, their gradient in mesh units,
   and log10(f - f_opt) per grid point for the readout. */
function field(f, c, N, scale) {
  const n = c.length;
  const step = (f.log_hi - f.log_lo) / LEVELS;
  const logv = new Float32Array(n);
  for (let i = 0; i < n; i++) logv[i] = c[i] ? f.log_lo + (c[i] - 1) * step : -Infinity;
  const h = new Float32Array(n);
  let lo, hi;
  if (scale === "log") {
    const sorted = Uint16Array.from(c).sort();
    let first = 0;
    while (first < n && sorted[first] === 0) first++;
    const q = sorted[Math.min(n - 1, first + Math.floor(0.005 * (n - first)))];
    hi = f.log_hi;
    lo = Math.max(f.log_lo + (q - 1) * step, hi - 8);
    const inv = 1 / (hi - lo);
    for (let i = 0; i < n; i++) h[i] = logv[i] <= lo ? 0 : (logv[i] - lo) * inv;
  } else {
    lo = 0; hi = Math.pow(10, f.log_hi);
    for (let i = 0; i < n; i++) h[i] = c[i] ? Math.pow(10, logv[i] - f.log_hi) : 0;
  }
  const g = new Float32Array(2 * n), cell = 2 / (N - 1);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const a = j * N + i;
    const il = i > 0 ? a - 1 : a, ir = i < N - 1 ? a + 1 : a;
    const jd = j > 0 ? a - N : a, ju = j < N - 1 ? a + N : a;
    g[2 * a] = HEIGHT * (h[ir] - h[il]) / ((ir - il) * cell);
    g[2 * a + 1] = HEIGHT * (h[ju] - h[jd]) / (((ju - jd) / N) * cell);
  }
  return { h, g, lo, hi, logv, scale };
}

/* ---------- colour maps: 256 RGBA entries interpolated between hex stops ---------- */
function ramp(stops) {
  const lut = new Uint8Array(256 * 4);
  const rgb = stops.map(h => [1, 3, 5].map(o => parseInt(h.slice(o, o + 2), 16)));
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (rgb.length - 1), k = Math.min(rgb.length - 2, Math.floor(t)), u = t - k;
    for (let c = 0; c < 3; c++) lut[4 * i + c] = Math.round(rgb[k][c] * (1 - u) + rgb[k + 1][c] * u);
    lut[4 * i + 3] = 255;
  }
  return lut;
}
/* blue (low) to red (high), Moreland's cool-warm */
const COOLWARM = [
  "#3b4cc0", "#445acc", "#4d68d7", "#5775e1", "#6282ea", "#6c8ef1", "#779af7", "#82a5fb", "#8db0fe",
  "#98b9ff", "#a3c2ff", "#aec9fd", "#b8d0f9", "#c2d5f4", "#ccd9ee", "#d5dbe6", "#dddddd", "#e5d8d1",
  "#ecd3c5", "#f1ccb9", "#f5c4ad", "#f7bba0", "#f7b194", "#f7a687", "#f49a7b", "#f18d6f", "#ec7f63",
  "#e57059", "#de604d", "#d55042", "#cb3e38", "#c0282f", "#b40426",
];
const LUT = ramp(COOLWARM);

/* ---------- formatting ---------- */
const neg = s => String(s).replace(/-/g, "−");
const sup = s => `<sup>${neg(s)}</sup>`;
function num(v, digits = 3) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-2) {
    const e = Math.floor(Math.log10(a)), m = v / Math.pow(10, e);
    return `${neg(m.toFixed(digits - 1))}×10${sup(e)}`;
  }
  return neg(Number(v.toPrecision(digits)).toString());
}
function coord(v) { return neg(v.toFixed(2)); }

const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

/* ---------- the stage ---------- */
function create(opts) {
  const N = opts.N, LO = opts.lo, HI = opts.hi;
  const view = opts.container, axes = opts.axes;
  const themeEl = opts.themeElement || document.documentElement;
  const pinVar = opts.pinVar || "--accent";
  const reduceMotion = !!opts.reduceMotion;
  const now = () => performance.now();
  const S = {
    view: opts.view || "elevation", angle: opts.angle ?? -0.55, tilt: TILT_ELEV, zoom: 1,
    current: null, dragging: false, pinOn: true, renderer: null, contour: opts.contour ?? 1,
  };
  const anims = {};
  function animate(name, to, ms) {
    const from = anims[name] ? value(name) : to;
    anims[name] = { from, to, t0: now(), ms: reduceMotion ? 0 : ms };
  }
  function value(name) {
    const a = anims[name];
    if (!a) return 0;
    const t = a.ms ? Math.min(1, (now() - a.t0) / a.ms) : 1;
    return a.from + (a.to - a.from) * ease(t);
  }
  anims.morph = { from: 1, to: 1, t0: 0, ms: 0 };
  anims.tilt = { from: TILT_ELEV, to: TILT_ELEV, t0: 0, ms: 0 };
  anims.mix = { from: 1, to: 1, t0: 0, ms: 0 };

  const canvas = document.createElement("canvas");
  canvas.className = "gl";
  canvas.setAttribute("aria-label", opts.label || "Rotating landscape; drag to turn, scroll to zoom");
  view.prepend(canvas);
  let overlay = null, octx = null;
  if (opts.overlay) {
    overlay = document.createElement("canvas");
    overlay.className = "overlay";
    overlay.setAttribute("aria-hidden", "true");
    canvas.after(overlay);
    octx = overlay.getContext("2d");
  }

  let renderer = null;
  try {
    if (root.THREE) renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  } catch (e) { renderer = null; }
  S.renderer = renderer;

  let scene, camera, world, mesh, uniforms, attrs, frame, pin, ring, head, lutTex;
  let ctx2d = null, image2d = null, lut = opts.lut || LUT;

  if (renderer) {
    renderer.setPixelRatio(Math.min(root.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(24, 1, 0.01, 80);
    camera.up.set(0, 0, 1);
    world = new THREE.Group();
    scene.add(world);

    const n = N * N, pos = new Float32Array(3 * n);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const a = j * N + i;
      pos[3 * a] = -1 + 2 * i / (N - 1); pos[3 * a + 1] = -1 + 2 * j / (N - 1);
    }
    const idx = new Uint16Array(6 * (N - 1) * (N - 1));
    let p = 0;
    for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
      const a = j * N + i, b = a + 1, c = a + N, d = c + 1;
      idx[p++] = a; idx[p++] = b; idx[p++] = d; idx[p++] = a; idx[p++] = d; idx[p++] = c;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    attrs = {
      aH0: new THREE.BufferAttribute(new Float32Array(n), 1),
      aH1: new THREE.BufferAttribute(new Float32Array(n), 1),
      aG0: new THREE.BufferAttribute(new Float32Array(2 * n), 2),
      aG1: new THREE.BufferAttribute(new Float32Array(2 * n), 2),
    };
    for (const [name, at] of Object.entries(attrs)) { at.setUsage(THREE.DynamicDrawUsage); geo.setAttribute(name, at); }

    lutTex = new THREE.DataTexture(lut, 256, 1, THREE.RGBAFormat);
    lutTex.magFilter = lutTex.minFilter = THREE.LinearFilter;
    lutTex.generateMipmaps = false;
    lutTex.needsUpdate = true;

    uniforms = {
      uMap: { value: lutTex }, uMix: { value: 1 }, uMorph: { value: 1 }, uHeight: { value: HEIGHT },
      uLevels: { value: 14 }, uContour: { value: .35 },
    };
    const material = new THREE.ShaderMaterial({
      uniforms, side: THREE.DoubleSide, extensions: { derivatives: true },
      vertexShader: `
        attribute float aH0; attribute float aH1; attribute vec2 aG0; attribute vec2 aG1;
        uniform float uMix; uniform float uMorph; uniform float uHeight;
        varying float vH; varying vec3 vN;
        void main() {
          float h = mix(aH0, aH1, uMix);
          vec2 g = mix(aG0, aG1, uMix) * uMorph;
          vH = h;
          vN = normalize(normalMatrix * normalize(vec3(-g, 1.0)));
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xy, h * uHeight * uMorph, 1.0);
        }`,
      fragmentShader: `
        uniform sampler2D uMap; uniform float uMorph; uniform float uLevels; uniform float uContour;
        varying float vH; varying vec3 vN;
        void main() {
          float h = clamp(vH, 0.0, 1.0);
          vec3 col = texture2D(uMap, vec2(0.002 + 0.996 * h, 0.5)).rgb;
          float t = h * uLevels;
          float w = fwidth(t);
          float d = abs(fract(t - 0.5) - 0.5);
          float line = (1.0 - smoothstep(0.0, 1.3 * w + 1e-4, d)) * (1.0 - smoothstep(0.12, 0.4, w));
          float lum = dot(col, vec3(0.299, 0.587, 0.114));
          vec3 ink = lum < 0.5 ? mix(col, vec3(1.0), 0.7) : col * 0.38;   // light lines on dark colours
          col = mix(col, ink, line * uContour);
          vec3 n = normalize(vN);
          if (!gl_FrontFacing) n = -n;
          float lambert = max(dot(n, normalize(vec3(-0.35, 0.62, 0.70))), 0.0);
          col *= mix(1.0, 0.48 + 0.62 * lambert, uMorph);
          gl_FragColor = vec4(col, 1.0);
        }`,
    });
    mesh = new THREE.Mesh(geo, material);
    mesh.frustumCulled = false;
    world.add(mesh);

    // the square of the domain with ticks at -5, -2.5, 0, 2.5, 5
    const fpts = [-1.0, -1.0, 0, 1.0, -1.0, 0, 1.0, -1.0, 0, 1.0, 1.0, 0, 1.0, 1.0, 0, -1.0, 1.0, 0, -1.0, 1.0, 0, -1.0, -1.0, 0];
    for (const u of [-1, -0.5, 0, 0.5, 1]) {
      fpts.push(u, -1, 0, u, -1.05, 0, u, 1, 0, u, 1.05, 0, -1, u, 0, -1.05, u, 0, 1, u, 0, 1.05, u, 0);
    }
    const fgeo = new THREE.BufferGeometry();
    fgeo.setAttribute("position", new THREE.Float32BufferAttribute(fpts, 3));
    frame = new THREE.LineSegments(fgeo, new THREE.LineBasicMaterial({ transparent: true, opacity: .8 }));
    world.add(frame);

    // the optimum: a ring on the surface, a stem and a head above the landscape
    const accent = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, depthTest: false, transparent: true });
    ring = new THREE.Mesh(new THREE.RingGeometry(0.03, 0.045, 40), accent);
    ring.renderOrder = 3;
    head = new THREE.Mesh(new THREE.SphereGeometry(0.022, 20, 12), new THREE.MeshBasicMaterial());
    const sgeo = new THREE.BufferGeometry();
    sgeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    const stem = new THREE.Line(sgeo, new THREE.LineBasicMaterial());
    pin = new THREE.Group();
    pin.add(ring, head, stem);
    pin.userData.stem = stem;
    world.add(pin);
  } else {
    ctx2d = canvas.getContext("2d");
    image2d = document.createElement("canvas");
    image2d.width = image2d.height = N;
    S.view = "heat";
    if (opts.onNoGL) opts.onNoGL();
  }
  if (S.view === "heat") {
    anims.morph = { from: 0, to: 0, t0: 0, ms: 0 };
    anims.tilt = { from: TILT_TOP, to: TILT_TOP, t0: 0, ms: 0 };
  }

  /* theme colours for the line work */
  function themeColours() {
    if (!renderer) return;
    const cs = getComputedStyle(themeEl);
    frame.material.color.set(cs.getPropertyValue("--muted").trim() || "#566874");
    const accent = cs.getPropertyValue(pinVar).trim() || cs.getPropertyValue("--accent").trim() || "#d55e00";
    ring.material.color.set(accent); head.material.color.set(accent); pin.userData.stem.material.color.set(accent);
  }
  themeColours();
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", themeColours);
  new MutationObserver(themeColours).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  /* ---------- axis labels ---------- */
  const LABELS = !axes ? [] : [
    [-1, -1.14, "−5"], [0, -1.14, "0"], [1, -1.14, "5"], [0, -1.3, "x₁", true],
    [1.14, -1, "−5"], [1.14, 0, "0"], [1.14, 1, "5"], [1.3, 0, "x₂", true],
  ].map(([x, y, text, name]) => {
    const el = document.createElement("span");
    el.textContent = text;
    if (name) el.className = "name";
    axes.appendChild(el);
    return { x, y, el };
  });

  /* ---------- field, colour map, view, pin ---------- */
  function setField(next, blend) {
    if (renderer) {
      if (S.current) { attrs.aH0.array.set(S.current.h); attrs.aG0.array.set(S.current.g); }
      else { attrs.aH0.array.set(next.h); attrs.aG0.array.set(next.g); }
      attrs.aH1.array.set(next.h); attrs.aG1.array.set(next.g);
      for (const at of Object.values(attrs)) at.needsUpdate = true;
      anims.mix = { from: 0, to: 1, t0: now(), ms: blend && !reduceMotion ? 700 : 0 };
    }
    S.current = next;
    if (ctx2d) paint2d();
  }
  function setLut(next) {
    lut = next;
    if (renderer) { lutTex.image.data = next; lutTex.needsUpdate = true; }
    if (ctx2d && S.current) paint2d();
  }
  function setView(v) {
    if (v === "elevation" && !renderer) return false;
    S.view = v;
    animate("morph", v === "elevation" ? 1 : 0, 900);
    animate("tilt", v === "elevation" ? S.tilt : TILT_TOP, 900);
    return true;
  }
  let pinXY = null;
  function setPin(xy) {
    pinXY = xy;
    if (pin && xy) pin.position.set(toMesh(xy[0]), toMesh(xy[1]), 0);
    if (pin) pin.visible = S.pinOn && !!xy;
  }
  function showPin(on) {
    S.pinOn = on;
    if (pin) pin.visible = on && !!pinXY;
  }

  /* ---------- geometry helpers ---------- */
  function heightAt(x, y) {   // bilinear, x and y in mesh units [-1, 1]
    const h = S.current.h;
    const fx = Math.min(N - 1.0001, Math.max(0, (x + 1) / 2 * (N - 1)));
    const fy = Math.min(N - 1.0001, Math.max(0, (y + 1) / 2 * (N - 1)));
    const i = Math.floor(fx), j = Math.floor(fy), u = fx - i, v = fy - j, a = j * N + i;
    return (h[a] * (1 - u) + h[a + 1] * u) * (1 - v) + (h[a + N] * (1 - u) + h[a + N + 1] * u) * v;
  }
  const toMesh = v => (v - LO) / (HI - LO) * 2 - 1;

  /* ---------- camera ---------- */
  const rim = [];
  for (let a = 0; a < 24; a++) rim.push([Math.cos(a / 24 * 2 * Math.PI) * 1.48, Math.sin(a / 24 * 2 * Math.PI) * 1.48]);
  function placeCamera(width, height) {
    const morph = value("morph"), tilt = value("tilt");
    const top = HEIGHT * morph, tz = 0.42 * top;
    const vf = camera.fov * Math.PI / 180, tanV = Math.tan(vf / 2), tanH = tanV * width / height;
    const fy = Math.cos(tilt), fz = -Math.sin(tilt);       // forward = (0, fy, fz)
    const uy = Math.sin(tilt), uz = Math.cos(tilt);        // up      = (0, uy, uz)
    let d = 0;
    for (const [x, y] of rim) for (const z of [0, top]) {
      const dz = z - tz, w = y * fy + dz * fz, v = y * uy + dz * uz;
      d = Math.max(d, w + Math.abs(x) / tanH, w + Math.abs(v) / tanV);
    }
    d *= 1.03 * S.zoom;
    camera.aspect = width / height;
    camera.position.set(0, -d * fy, tz - d * fz);
    camera.lookAt(0, 0, tz);
    camera.updateProjectionMatrix();
  }

  /* ---------- 2-D fallback ---------- */
  function paint2d() {
    const ictx = image2d.getContext("2d"), img = ictx.createImageData(N, N), h = S.current.h;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const t = Math.min(255, Math.max(0, Math.round(h[(N - 1 - r) * N + c] * 255))), o = 4 * (r * N + c);
      img.data[o] = lut[4 * t]; img.data[o + 1] = lut[4 * t + 1]; img.data[o + 2] = lut[4 * t + 2]; img.data[o + 3] = 255;
    }
    ictx.putImageData(img, 0, 0);
  }
  function draw2d(w, h, dpr) {
    const cw = Math.round(w * dpr), ch = Math.round(h * dpr);
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const s = Math.min(w, h) / (2 * 1.5) / S.zoom;
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx2d.clearRect(0, 0, w, h);
    ctx2d.translate(w / 2, h / 2); ctx2d.rotate(-S.angle); ctx2d.scale(s, s);
    ctx2d.imageSmoothingEnabled = true;
    ctx2d.drawImage(image2d, -1, -1, 2, 2);
    if (S.pinOn && pinXY) {
      ctx2d.beginPath();
      ctx2d.arc(toMesh(pinXY[0]), -toMesh(pinXY[1]), 0.04, 0, 2 * Math.PI);
      ctx2d.lineWidth = 0.015;
      const cs = getComputedStyle(themeEl);
      ctx2d.strokeStyle = cs.getPropertyValue(pinVar).trim() || cs.getPropertyValue("--accent");
      ctx2d.stroke();
    }
    for (const L of LABELS) {
      const c = Math.cos(-S.angle), sn = Math.sin(-S.angle), x = L.x, y = -L.y;
      L.el.style.left = `${w / 2 + s * (c * x - sn * y)}px`;
      L.el.style.top = `${h / 2 + s * (sn * x + c * y)}px`;
    }
  }

  /* screen position (CSS px in the container) of the domain point (x1, x2) on the surface */
  const tmp = renderer ? new THREE.Vector3() : null;
  let size = [0, 0];
  function project(x1, x2) {
    const mx = toMesh(x1), my = toMesh(x2), [w, h] = size;
    if (renderer) {
      const z = HEIGHT * value("morph") * heightAt(Math.max(-1, Math.min(1, mx)), Math.max(-1, Math.min(1, my)));
      tmp.set(mx, my, z + 0.004).applyMatrix4(world.matrixWorld).project(camera);
      return [(tmp.x + 1) / 2 * w, (1 - tmp.y) / 2 * h];
    }
    const s = Math.min(w, h) / (2 * 1.5) / S.zoom;
    const c = Math.cos(-S.angle), sn = Math.sin(-S.angle), y = -my;
    return [w / 2 + s * (c * mx - sn * y), h / 2 + s * (sn * mx + c * y)];
  }

  /* ---------- interaction: drag to turn and tilt, wheel to zoom ---------- */
  let drag = null;
  canvas.addEventListener("pointerdown", e => {
    drag = { x: e.clientX, y: e.clientY, moved: 0, id: e.pointerId };
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add("dragging");
    S.dragging = true;
  });
  canvas.addEventListener("pointermove", e => {
    if (drag && drag.id === e.pointerId) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.moved += Math.abs(dx) + Math.abs(dy);
      drag.x = e.clientX; drag.y = e.clientY;
      S.angle -= dx * 0.009;
      if (S.view === "elevation" && renderer) {
        S.tilt = Math.min(TILT_TOP, Math.max(10 * Math.PI / 180, S.tilt + dy * 0.006));
        anims.tilt = { from: S.tilt, to: S.tilt, t0: 0, ms: 0 };
      }
      if (drag.moved > 4 && opts.onDrag) opts.onDrag();
    } else if (e.pointerType === "mouse" && opts.onPoint) {
      opts.onPoint(e.clientX, e.clientY);
    }
  });
  function endDrag(e) {
    if (!drag) return;
    if (drag.moved <= 4 && opts.onPoint) opts.onPoint(e.clientX, e.clientY);
    drag = null;
    S.dragging = false;
    canvas.classList.remove("dragging");
  }
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", () => { drag = null; S.dragging = false; canvas.classList.remove("dragging"); });
  canvas.addEventListener("pointerleave", e => { if (!drag && e.pointerType === "mouse" && opts.onLeave) opts.onLeave(); });
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    S.zoom = Math.min(2.2, Math.max(0.45, S.zoom * Math.exp(e.deltaY * 0.0012)));
  }, { passive: false });

  /* the domain point under the pointer, by marching the view ray over the surface (mesh units) */
  const raycaster = renderer ? new THREE.Raycaster() : null;
  function pick(cx, cy) {
    const r = canvas.getBoundingClientRect();
    const nx = (cx - r.left) / r.width * 2 - 1, ny = -(cy - r.top) / r.height * 2 + 1;
    if (!renderer) {
      const s = Math.min(r.width, r.height) / (2 * 1.5) / S.zoom;
      const px = (cx - r.left - r.width / 2) / s, py = (cy - r.top - r.height / 2) / s;
      const c = Math.cos(S.angle), sn = Math.sin(S.angle);
      return [c * px - sn * py, -(sn * px + c * py)];
    }
    raycaster.setFromCamera({ x: nx, y: ny }, camera);
    const inv = world.matrixWorld.clone().invert();
    const o = raycaster.ray.origin.clone().applyMatrix4(inv);
    const dir = raycaster.ray.direction.clone().transformDirection(inv);
    const top = HEIGHT * value("morph");
    if (top < 1e-3) {
      const t = -o.z / dir.z;
      return t > 0 ? [o.x + t * dir.x, o.y + t * dir.y] : null;
    }
    let t0 = 0, t1 = 50;
    for (const [oc, dc, lo, hi] of [[o.x, dir.x, -1, 1], [o.y, dir.y, -1, 1], [o.z, dir.z, 0, top]]) {
      if (Math.abs(dc) < 1e-12) { if (oc < lo || oc > hi) return null; continue; }
      let a = (lo - oc) / dc, b = (hi - oc) / dc;
      if (a > b) [a, b] = [b, a];
      t0 = Math.max(t0, a); t1 = Math.min(t1, b);
    }
    if (t0 > t1) return null;
    const gap = t => o.z + t * dir.z - top * heightAt(o.x + t * dir.x, o.y + t * dir.y);
    const steps = 360;
    let prev = t0;
    if (gap(t0) <= 0) return [o.x + t0 * dir.x, o.y + t0 * dir.y];
    for (let s = 1; s <= steps; s++) {
      const t = t0 + (t1 - t0) * s / steps;
      if (gap(t) <= 0) {
        let a = prev, b = t;
        for (let it = 0; it < 18; it++) { const m = (a + b) / 2; if (gap(m) > 0) a = m; else b = m; }
        return [o.x + b * dir.x, o.y + b * dir.y];
      }
      prev = t;
    }
    return null;
  }

  /* ---------- one frame ---------- */
  function draw() {
    const w = view.clientWidth, h = view.clientHeight;
    const dpr = Math.min(root.devicePixelRatio || 1, 2);
    if (renderer) {
      if (w !== size[0] || h !== size[1]) { renderer.setSize(w, h, false); size = [w, h]; }
      const morph = value("morph");
      uniforms.uMorph.value = morph;
      uniforms.uMix.value = value("mix");
      uniforms.uContour.value = (0.62 - 0.24 * morph) * S.contour;
      world.rotation.z = S.angle;
      placeCamera(w || 1, h || 1);
      if (pin.visible) {
        const zs = HEIGHT * morph * heightAt(pin.position.x, pin.position.y);
        const ztop = zs + (HEIGHT * morph + 0.12 - zs) * morph;
        ring.position.z = zs + 0.004;
        head.position.z = ztop;
        head.visible = morph > 0.05;
        const sp = pin.userData.stem.geometry.attributes.position;
        sp.setXYZ(0, 0, 0, zs); sp.setXYZ(1, 0, 0, ztop); sp.needsUpdate = true;
      }
      world.updateMatrixWorld();
      renderer.render(scene, camera);
      const v = new THREE.Vector3();
      for (const L of LABELS) {
        v.set(L.x, L.y, 0).applyMatrix4(world.matrixWorld).project(camera);
        L.el.style.left = `${(v.x + 1) / 2 * w}px`;
        L.el.style.top = `${(1 - v.y) / 2 * h}px`;
        L.el.hidden = v.z > 1;
      }
    } else {
      size = [w, h];
      draw2d(w, h, dpr);
    }
    if (overlay) {
      const cw = Math.round(w * dpr), ch = Math.round(h * dpr);
      if (overlay.width !== cw || overlay.height !== ch) { overlay.width = cw; overlay.height = ch; }
      octx.setTransform(dpr, 0, 0, dpr, 0, 0);
      octx.clearRect(0, 0, w, h);
      opts.overlay(octx, project, w, h);
    }
  }

  return {
    state: S, canvas, setField, setLut, setView, setPin, showPin, pick, project, draw,
    morph: () => value("morph"),
    get current() { return S.current; },
  };
}

root.BBOBLandscape = { decode, field, ramp, COOLWARM, LUT, HEIGHT, LEVELS, num, neg, sup, coord, create };
})(window);
