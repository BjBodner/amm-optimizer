/* Recorded-run player (RunPlayer.astro). Vanilla JS, no dependencies, no storage, no autoplay.
   Data: /demo/<run>.json written by site/tools/record_run.py. Keys on the focused player:
   Space play/pause, Left/Right step (Shift: 10), Home/End jump, R reset. State in the URL hash:
   #demo=f15&e=120 (function and evaluation). */
(function () {
	'use strict';
	// Run data sits next to this script, so the player works under any base path.
	var DEMO_BASE = document.currentScript && document.currentScript.src ? document.currentScript.src.replace(/[^/]*$/, '') : '/demo/';
	var root = document.querySelector('[data-rp]');
	if (!root) return;
	var runs = JSON.parse(root.getAttribute('data-runs'));
	var $ = function (s) { return root.querySelector(s); };
	var els = {
		bar: $('.rp-bar'), app: $('.rp-app'), fn: $('[data-fn]'), speed: $('[data-speed]'), scrub: $('[data-scrub]'),
		play: $('[data-act="play"]'), status: $('[data-status]'), live: $('[data-live]'), error: $('[data-error]'),
		strip: $('[data-strip]'), stripBar: $('[data-strip-bar]'), repro: $('[data-repro]'), focus: $('[data-focus]'),
		bestAmm: $('[data-best-amm]'), bestIpop: $('[data-best-ipop]')
	};
	var cv = { amm: $('[data-cv="amm"]'), ipop: $('[data-cv="ipop"]'), conv: $('[data-cv="conv"]') };
	var cache = {};
	var data = null, t = 0, N = 0, playing = false, last = 0, acc = 0, band = null, bestIdx = null;

	function parseHash() {
		var o = {};
		(location.hash || '').replace(/^#/, '').split('&').forEach(function (p) {
			var kv = p.split('=');
			if (kv.length === 2) o[kv[0]] = decodeURIComponent(kv[1]);
		});
		return o;
	}
	function writeHash() {
		if (!data || !window.history || !history.replaceState) return;
		try { history.replaceState(null, '', '#demo=' + els.fn.value + '&e=' + t); } catch (e) { /* ignore */ }
	}

	function isDark() {
		var th = document.documentElement.getAttribute('data-theme');
		if (th === 'dark') return true;
		if (th === 'light') return false;
		return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
	}
	function css(name, fallback) {
		var v = getComputedStyle(root).getPropertyValue(name).trim();
		return v || fallback;
	}
	function fmt(r) {
		if (r === undefined || r === null) return 'n/a';
		if (r <= 1e-12) return '≤ 1e-12';
		return r >= 1e-3 && r < 1e4 ? r.toPrecision(3) : r.toExponential(1);
	}

	/* ---- derived series ---- */
	function bestSoFar(run) {
		var out = new Array(run.n), b = Infinity, idx = new Array(run.n), bi = 0;
		for (var i = 0; i < run.n; i++) {
			if (run.regret[i] < b) { b = run.regret[i]; bi = i; }
			out[i] = b; idx[i] = bi;
		}
		run.best = out; run.bestIdx = idx;
	}

	/* ---- landscape image, bilinear on the grid, quantised to half decades so bands read as contours ---- */
	function ramp(level, nLevels, dark) {
		var u = nLevels > 1 ? level / (nLevels - 1) : 0; /* 0 = near the optimum value, 1 = far */
		var a = dark ? [46, 98, 140] : [150, 196, 224];
		var b = dark ? [16, 20, 24] : [250, 250, 247];
		return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
	}
	function buildBand(canvas) {
		var g = data.grid, n = g.n, w = canvas.width, h = canvas.height, dark = isDark();
		var vmin = Infinity, vmax = -Infinity;
		for (var i = 0; i < g.v.length; i++) { if (g.v[i] < vmin) vmin = g.v[i]; if (g.v[i] > vmax) vmax = g.v[i]; }
		var step = 5, nLev = Math.max(2, Math.ceil((vmax - vmin) / step) + 1);
		var off = document.createElement('canvas'); off.width = w; off.height = h;
		var c = off.getContext('2d'), img = c.createImageData(w, h), d = img.data;
		for (var py = 0; py < h; py++) {
			var gy = (1 - py / (h - 1)) * (n - 1), y0 = Math.min(n - 2, Math.floor(gy)), fy = gy - y0;
			for (var px = 0; px < w; px++) {
				var gx = (px / (w - 1)) * (n - 1), x0 = Math.min(n - 2, Math.floor(gx)), fx = gx - x0;
				var v00 = g.v[y0 * n + x0], v10 = g.v[y0 * n + x0 + 1], v01 = g.v[(y0 + 1) * n + x0], v11 = g.v[(y0 + 1) * n + x0 + 1];
				var v = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
				var lev = Math.floor((v - vmin) / step), col = ramp(lev, nLev, dark), k = (py * w + px) * 4;
				d[k] = col[0]; d[k + 1] = col[1]; d[k + 2] = col[2]; d[k + 3] = 255;
			}
		}
		c.putImageData(img, 0, 0);
		return off;
	}

	function sizeCanvas(c, ratio) {
		var dpr = Math.min(window.devicePixelRatio || 1, 2), w = Math.max(160, Math.round(c.clientWidth * dpr));
		if (c.width !== w) { c.width = w; c.height = Math.round(w / ratio); return true; }
		return false;
	}

	/* ---- drawing ---- */
	function drawPanel(key) {
		var c = cv[key];
		if (sizeCanvas(c, 1) || !band || band.w !== c.width || band.dark !== isDark() || band.fn !== els.fn.value) {
			band = { img: buildBand(c), w: c.width, dark: isDark(), fn: els.fn.value };
		}
		var ctx = c.getContext('2d'), W = c.width, run = data.runs[key];
		ctx.drawImage(band.img, 0, 0);
		var lo = data.grid.lo, hi = data.grid.hi;
		function X(v) { return (v - lo) / (hi - lo) * (W - 1); }
		function Y(v) { return (1 - (v - lo) / (hi - lo)) * (W - 1); }
		var colour = css(key === 'amm' ? '--m-amm' : '--m-ipop', key === 'amm' ? '#111' : '#009e73');
		var halo = isDark() ? 'rgba(0,0,0,0.65)' : 'rgba(255,255,255,0.8)';
		var r = Math.max(2, W / 150), n = Math.min(t, run.n);
		ctx.lineWidth = 1;
		for (var i = 0; i < n; i++) {
			var x = X(run.x[2 * i]), y = Y(run.x[2 * i + 1]), recent = n - i <= 12;
			ctx.globalAlpha = recent ? 1 : 0.55;
			ctx.fillStyle = colour; ctx.strokeStyle = halo;
			ctx.beginPath();
			if (key === 'amm') { ctx.arc(x, y, recent ? r * 1.3 : r, 0, 6.2832); }
			else { var s = recent ? r * 1.3 : r; ctx.rect(x - s, y - s, 2 * s, 2 * s); }
			ctx.fill(); ctx.stroke();
		}
		ctx.globalAlpha = 1;
		if (n > 0) {
			var b = run.bestIdx[n - 1], bx = X(run.x[2 * b]), by = Y(run.x[2 * b + 1]), R = r * 3.2;
			ctx.beginPath();
			for (var k = 0; k < 10; k++) {
				var ang = -Math.PI / 2 + k * Math.PI / 5, rad = k % 2 ? R * 0.45 : R;
				ctx[k ? 'lineTo' : 'moveTo'](bx + rad * Math.cos(ang), by + rad * Math.sin(ang));
			}
			ctx.closePath();
			ctx.fillStyle = css('--oi-vermillion', '#d55e00'); ctx.fill();
			ctx.lineWidth = 1.5; ctx.strokeStyle = isDark() ? '#fff' : '#000'; ctx.stroke();
		}
		ctx.strokeStyle = isDark() ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.45)';
		ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, W - 1, c.height - 1);
	}

	function drawConv() {
		var c = cv.conv; sizeCanvas(c, 16 / 7);
		var ctx = c.getContext('2d'), W = c.width, H = c.height, dark = isDark();
		var ink = dark ? '#e8e8e8' : '#222', grid = dark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.12)';
		var fs = Math.max(10, Math.round(W / 62));
		var padL = fs * 3.4, padR = fs * 0.8, padT = fs * 0.8, padB = fs * 2.2;
		function lg(v) { return Math.log10(Math.max(v, 1e-12)); }
		var ymin = 0, ymax = -Infinity;
		['amm', 'ipop'].forEach(function (k) {
			var run = data.runs[k];
			for (var i = 0; i < run.n; i++) { var l = lg(run.regret[i]); if (l > ymax) ymax = l; if (l < ymin) ymin = l; }
		});
		ymin = Math.max(-12, Math.floor(ymin)); ymax = Math.ceil(ymax);
		if (ymax <= ymin) ymax = ymin + 1;
		function X(i) { return padL + (i / N) * (W - padL - padR); }
		function Y(l) { return padT + (ymax - l) / (ymax - ymin) * (H - padT - padB); }
		ctx.clearRect(0, 0, W, H);
		ctx.font = fs + 'px system-ui, sans-serif'; ctx.fillStyle = ink; ctx.textBaseline = 'middle';
		var every = Math.ceil((ymax - ymin) / 6);
		for (var p = ymin; p <= ymax; p += every) {
			ctx.strokeStyle = grid; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(padL, Y(p)); ctx.lineTo(W - padR, Y(p)); ctx.stroke();
			ctx.textAlign = 'right'; ctx.fillText('1e' + p, padL - 4, Y(p));
		}
		ctx.textAlign = 'center'; ctx.textBaseline = 'top';
		[0, 0.25, 0.5, 0.75, 1].forEach(function (f) { ctx.fillText(String(Math.round(f * N)), X(f * N), H - padB + 4); });
		/* ATM span shading, only from the documented hook */
		var spans = data.header.stages && data.header.stages.available ? data.header.stages.atm_sampler_spans : [];
		spans.forEach(function (s) {
			ctx.fillStyle = dark ? 'rgba(230,159,0,0.22)' : 'rgba(230,159,0,0.25)';
			ctx.fillRect(X(s[0]), padT, X(s[1]) - X(s[0]), H - padT - padB);
		});
		function line(key, colour, width, dash) {
			var run = data.runs[key], n = Math.min(t, run.n);
			if (n < 1) return;
			ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.setLineDash(dash);
			ctx.beginPath();
			for (var i = 0; i < n; i++) { var x = X(i), y = Y(lg(run.best[i])); if (i === 0) ctx.moveTo(x, y); else { ctx.lineTo(x, y); } }
			ctx.stroke(); ctx.setLineDash([]);
		}
		line('ipop', css('--m-ipop', '#009e73'), Math.max(2, W / 300), [fs * 0.7, fs * 0.45]);
		line('amm', css('--m-amm', '#111'), Math.max(2.5, W / 230), []);
		ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.setLineDash([]);
		ctx.beginPath(); ctx.moveTo(X(t), padT); ctx.lineTo(X(t), H - padB); ctx.stroke();
		ctx.strokeRect(padL + 0.5, padT + 0.5, W - padL - padR, H - padT - padB);
		ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = ink;
		ctx.fillText('evaluations', padL + (W - padL - padR) / 2, H - 1);
	}

	function stageText() {
		var st = data.header.stages;
		if (!st || !st.available) return 'AMM stage: no stage data for this run (the frozen code reports none for the stage AMM used).';
		var inside = st.atm_sampler_spans.some(function (s) { return t > s[0] && t <= s[1]; });
		return inside ? 'AMM stage: the ATM sampler is running (reported by the profiler hook).' : 'AMM stage: not reported by any hook (not the ATM sampler).';
	}
	function describe() {
		var a = t ? fmt(data.runs.amm.best[Math.min(t, data.runs.amm.n) - 1]) : 'n/a';
		var b = t ? fmt(data.runs.ipop.best[Math.min(t, data.runs.ipop.n) - 1]) : 'n/a';
		return { a: a, b: b, text: 'Evaluation ' + t + ' of ' + N + '. Best regret so far: AMM ' + a + ', pycma IPOP ' + b + '. ' + stageText() };
	}
	function render(announce) {
		drawPanel('amm'); drawPanel('ipop'); drawConv();
		var d = describe();
		els.status.textContent = d.text;
		els.bestAmm.textContent = 'best regret ' + d.a;
		els.bestIpop.textContent = 'best regret ' + d.b;
		els.scrub.value = String(t);
		els.scrub.setAttribute('aria-valuetext', 'Evaluation ' + t + ' of ' + N);
		var cur = els.stripBar.querySelector('.rp-cursor');
		if (cur) cur.style.left = (t / N * 100) + '%';
		if (announce) els.live.textContent = d.text;
	}

	/* ---- playback ---- */
	function setPlaying(on) {
		playing = on;
		els.play.textContent = on ? 'Pause' : 'Play';
		if (on) { last = 0; acc = 0; requestAnimationFrame(tick); } else if (data) { render(true); writeHash(); }
	}
	function tick(ts) {
		if (!playing) return;
		if (!last) last = ts;
		acc += (ts - last) / 1000 * Number(els.speed.value); last = ts;
		var whole = Math.floor(acc);
		if (whole > 0) { acc -= whole; t = Math.min(N, t + whole); render(false); }
		if (t >= N) { setPlaying(false); return; }
		requestAnimationFrame(tick);
	}
	function go(to, announce) { t = Math.max(0, Math.min(N, to)); render(announce !== false); writeHash(); }
	function togglePlay() {
		if (playing) { setPlaying(false); return; }
		if (t >= N) t = 0;
		setPlaying(true);
	}

	function buildStrip() {
		var st = data.header.stages, bar = els.stripBar;
		bar.textContent = '';
		if (!st || !st.available) { els.strip.hidden = true; return; }
		els.strip.hidden = false;
		st.atm_sampler_spans.forEach(function (s) {
			var seg = document.createElement('div');
			seg.className = 'rp-seg';
			seg.style.left = (s[0] / N * 100) + '%';
			seg.style.width = ((s[1] - s[0]) / N * 100) + '%';
			seg.textContent = 'ATM sampler';
			bar.appendChild(seg);
		});
		var cur = document.createElement('div'); cur.className = 'rp-cursor'; bar.appendChild(cur);
		bar.setAttribute('role', 'img');
		bar.setAttribute('aria-label', 'Timeline: the ATM sampler ran from evaluation ' + st.atm_sampler_spans.map(function (s) { return s[0] + ' to ' + s[1]; }).join(' and from ') + '. Other stages are not reported.');
	}
	function buildRepro() {
		var h = data.header, rows = [
			['function', h.function], ['dimension', String(h.D)], ['budget', h.budget + ' evaluations (' + h.budget_per_dim + ' x D)'],
			['seed', String(h.seed)], ['bbob instance', String(h.instance) + ' (f_opt ' + h.f_opt + ')'], ['start point', JSON.stringify(h.x0)],
			['code', 'commit ' + h.code_commit + ', preset ' + h.preset + ', config hash ' + h.config_hash],
			['AMM call', h.amm], ['pycma call', h.ipop], ['note', h.note_ipop],
			['versions', Object.keys(h.versions).map(function (k) { return k + ' ' + h.versions[k]; }).join(', ')],
			['stage data', h.stages.source + '; ' + h.stages.other_stages], ['recorded', h.recorded]
		];
		els.repro.textContent = '';
		rows.forEach(function (r) {
			var dt = document.createElement('dt'), dd = document.createElement('dd');
			dt.textContent = r[0]; dd.textContent = r[1];
			els.repro.appendChild(dt); els.repro.appendChild(dd);
		});
	}

	function load(id, e) {
		var meta = runs.filter(function (r) { return r.id === id; })[0] || runs[0];
		els.fn.value = meta.id;
		setPlaying(false);
		function done(d) {
			data = d; N = Math.min(d.header.budget, d.runs.amm.n); band = null;
			bestSoFar(d.runs.amm); bestSoFar(d.runs.ipop);
			els.scrub.max = String(N);
			t = (typeof e === 'number' && e >= 0) ? Math.min(N, e) : N; /* starts paused, at the end */
			buildStrip(); buildRepro();
			els.error.hidden = true;
			render(false);
		}
		if (cache[meta.file]) return done(cache[meta.file]);
		fetch(DEMO_BASE + meta.file).then(function (r) {
			if (!r.ok) throw new Error('HTTP ' + r.status);
			return r.json();
		}).then(function (d) { cache[meta.file] = d; done(d); }).catch(function (err) {
			els.error.hidden = false;
			els.error.textContent = 'The recorded run could not be loaded (' + err.message + ').';
		});
	}

	/* ---- events ---- */
	els.play.addEventListener('click', togglePlay);
	root.querySelector('[data-act="back"]').addEventListener('click', function () { setPlaying(false); go(t - 1); });
	root.querySelector('[data-act="fwd"]').addEventListener('click', function () { setPlaying(false); go(t + 1); });
	root.querySelector('[data-act="reset"]').addEventListener('click', function () { setPlaying(false); go(0); });
	els.scrub.addEventListener('input', function () { if (playing) setPlaying(false); t = Number(els.scrub.value); render(false); });
	els.scrub.addEventListener('change', function () { render(true); writeHash(); });
	els.fn.addEventListener('change', function () { load(els.fn.value); writeHash(); });
	root.addEventListener('keydown', function (ev) {
		var tag = ev.target && ev.target.tagName;
		if (tag === 'SELECT' || tag === 'INPUT' || ev.ctrlKey || ev.metaKey || ev.altKey || !data) return;
		var k = ev.key;
		if (k === ' ' || k === 'Spacebar') { if (tag === 'BUTTON' || tag === 'SUMMARY') return; ev.preventDefault(); togglePlay(); }
		else if (k === 'ArrowRight') { ev.preventDefault(); setPlaying(false); go(t + (ev.shiftKey ? 10 : 1)); }
		else if (k === 'ArrowLeft') { ev.preventDefault(); setPlaying(false); go(t - (ev.shiftKey ? 10 : 1)); }
		else if (k === 'Home') { ev.preventDefault(); setPlaying(false); go(0); }
		else if (k === 'End') { ev.preventDefault(); setPlaying(false); go(N); }
		else if (k === 'r' || k === 'R') { ev.preventDefault(); setPlaying(false); go(0); }
	});
	var resizeTimer = null;
	window.addEventListener('resize', function () { clearTimeout(resizeTimer); resizeTimer = setTimeout(function () { if (data) { band = null; render(false); } }, 120); });
	new MutationObserver(function () { if (data) { band = null; render(false); } }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
	if (window.matchMedia) {
		var mq = window.matchMedia('(prefers-color-scheme: dark)');
		var onScheme = function () { if (data) { band = null; render(false); } };
		if (mq.addEventListener) mq.addEventListener('change', onScheme);
	}

	window.addEventListener('hashchange', function () {
		var h = parseHash();
		if (!h.demo || !runs.some(function (r) { return r.id === h.demo; })) return;
		var e = h.e !== undefined ? Number(h.e) : undefined;
		if (h.demo === els.fn.value && data) { if (e !== undefined && e !== t) { setPlaying(false); go(e); } return; }
		load(h.demo, e);
	});

	/* ---- start: paused, nothing plays until Play is pressed ---- */
	els.bar.hidden = false; els.app.hidden = false; root.setAttribute('data-ready', '');
	var h = parseHash(), start = runs.filter(function (r) { return r.id === h.demo; })[0] ? h.demo : runs[0].id;
	load(start, h.e !== undefined ? Number(h.e) : undefined);
})();
