/* Minimal SVG charting, written by hand.
 *
 * A charting library would be a CDN request or a build step, and this dashboard
 * is meant to work offline from a clone. Everything here draws into an inline
 * <svg>: linear and logarithmic axes, gridlines, series lines, points, fitted
 * curves, reference lines and bars.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

/* Fallback only. The live palette comes from the active style's --c1..--c6
 * tokens so a theme owns its series colours as well as its chrome; Okabe-Ito
 * here keeps the default distinguishable under common colour blindness. */
export const FALLBACK_PALETTE = [
  "#0072B2", "#D55E00", "#009E73", "#CC79A7",
  "#E69F00", "#56B4E9", "#8C6BB1", "#B4453C",
];

/* Role tints, used where each mark carries its own label (bar charts).
 * Line charts must NOT colour by role: two baselines would share one hue and
 * become indistinguishable, which is a legibility bug rather than a semantic
 * grouping. There, every series gets its own palette entry and the role is
 * carried by the legend tag instead.
 */
export const ROLE_COLOURS = {
  baseline: "#D55E00",
  aegisq: "#0072B2",
  mitigated: "#009E73",
};

/* Resolve a CSS custom property to a concrete colour.
 *
 * SVG presentation attributes accept var() only patchily across browsers, and
 * the page is theme-aware, so colours are resolved at draw time instead of
 * being hardcoded for one theme. Drawing data in a dark-theme ink would make it
 * invisible on a light background, which is exactly the sort of failure a
 * screenshot in one theme never reveals.
 */
export function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export const ink = () => cssVar("--ink-line", cssVar("--text", "#1f2328"));
export const faint = () => cssVar("--faint", "#8c959f");

/* Series colours for the active style, resolved at draw time. */
export function palette() {
  const themed = [1, 2, 3, 4, 5, 6]
    .map((i) => cssVar(`--c${i}`, ""))
    .filter(Boolean);
  return themed.length ? themed : FALLBACK_PALETTE;
}

/* Kept as a named export for bar charts, whose marks carry their own labels. */
export const PALETTE = new Proxy({}, {
  get: (_, key) => {
    const list = palette();
    return typeof key === "string" && /^\d+$/.test(key) ? list[Number(key) % list.length] : undefined;
  },
});

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  if (parent) parent.appendChild(node);
  return node;
}

/* ---------------------------------------------------------------- scales */
function linearScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const scale = (v) => r0 + ((v - d0) / span) * (r1 - r0);
  scale.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * span;
  scale.domain = domain;
  scale.ticks = (count = 5) => niceTicks(d0, d1, count);
  return scale;
}

function logScale(domain, range) {
  const lo = Math.log10(Math.max(domain[0], Number.MIN_VALUE));
  const hi = Math.log10(Math.max(domain[1], Number.MIN_VALUE));
  const inner = linearScale([lo, hi], range);
  const scale = (v) => inner(Math.log10(Math.max(v, Number.MIN_VALUE)));
  scale.domain = domain;
  scale.ticks = () => {
    const out = [];
    for (let e = Math.floor(lo); e <= Math.ceil(hi); e += 1) out.push(10 ** e);
    return out.filter((t) => t >= 10 ** lo * 0.99 && t <= 10 ** hi * 1.01);
  };
  return scale;
}

function niceTicks(lo, hi, count) {
  if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
  const raw = (hi - lo) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalised = raw / magnitude;
  const step = (normalised >= 5 ? 10 : normalised >= 2 ? 5 : normalised >= 1 ? 2 : 1) * magnitude;
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 1e-6; v += step) ticks.push(Number(v.toFixed(12)));
  return ticks;
}

export function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || !isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e5)) {
    return value.toExponential(1).replace("e", "e");
  }
  return Number(value.toPrecision(digits)).toString();
}

export function formatPercent(value) {
  if (value === null || value === undefined || !isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
}

/* ----------------------------------------------------------------- chart */
/**
 * Draw a chart.
 *
 * spec = {
 *   series:  [{label, role?, colour?, points:[{x,y}], dashed?, showPoints?, width?}],
 *   bands:   [{label, points:[{x,y}], colour}]           // fitted curves
 *   markers: [{y, label, colour}]                        // horizontal references
 *   xLabel, yLabel, yScale: "linear"|"log", xTickFormat, yTickFormat,
 *   yDomain, xDomain, height
 * }
 */
export function drawChart(container, spec) {
  container.innerHTML = "";
  const width = container.clientWidth || 640;
  const height = spec.height || 300;
  const margin = { top: 14, right: 16, bottom: 40, left: 62 };
  const innerW = Math.max(width - margin.left - margin.right, 40);
  const innerH = Math.max(height - margin.top - margin.bottom, 40);

  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`, width: "100%", height,
    role: "img", "aria-label": spec.ariaLabel || spec.yLabel || "chart",
  }, container);
  const plot = el("g", { transform: `translate(${margin.left},${margin.top})` }, svg);

  const all = [...(spec.series || []), ...(spec.bands || [])];
  const xs = all.flatMap((s) => s.points.map((p) => p.x));
  // Error bars must fit inside the domain too, or a whisker gets clipped at
  // the axis -- which would hide exactly the spread the chart exists to show.
  let ys = all.flatMap((s) => s.points.flatMap((p) => {
    if (p.y === null || !isFinite(p.y)) return [];
    const err = p.err && isFinite(p.err) ? p.err : 0;
    return [p.y - err, p.y + err];
  }));
  (spec.markers || []).forEach((m) => ys.push(m.y));
  if (!xs.length || !ys.length) {
    el("text", { x: width / 2, y: height / 2, "text-anchor": "middle", class: "chart-empty" }, svg)
      .textContent = "no data";
    return;
  }

  const isLog = spec.yScale === "log";
  if (isLog) ys = ys.filter((v) => v > 0);

  let [y0, y1] = spec.yDomain || [Math.min(...ys), Math.max(...ys)];
  if (!spec.yDomain) {
    if (isLog) {
      y0 = 10 ** Math.floor(Math.log10(y0));
      y1 = 10 ** Math.ceil(Math.log10(y1));
    } else {
      const pad = (y1 - y0) * 0.12 || Math.abs(y1 || 1) * 0.12;
      y0 -= pad; y1 += pad;
    }
  }
  const [x0, x1] = spec.xDomain || [Math.min(...xs), Math.max(...xs)];

  const x = linearScale([x0, x1], [0, innerW]);
  const y = (isLog ? logScale : linearScale)([y0, y1], [innerH, 0]);

  /* grid + axes */
  const yTicks = y.ticks(5);
  for (const tick of yTicks) {
    const py = y(tick);
    if (!isFinite(py)) continue;
    el("line", { x1: 0, x2: innerW, y1: py, y2: py, class: "grid" }, plot);
    el("text", { x: -10, y: py + 4, "text-anchor": "end", class: "tick" }, plot)
      .textContent = spec.yTickFormat ? spec.yTickFormat(tick) : formatNumber(tick);
  }
  const xTicks = spec.xTicks || x.ticks(6);
  for (const tick of xTicks) {
    const px = x(tick);
    el("line", { x1: px, x2: px, y1: 0, y2: innerH, class: "grid grid-x" }, plot);
    el("text", { x: px, y: innerH + 20, "text-anchor": "middle", class: "tick" }, plot)
      .textContent = spec.xTickFormat ? spec.xTickFormat(tick) : formatNumber(tick);
  }
  el("line", { x1: 0, x2: innerW, y1: innerH, y2: innerH, class: "axis" }, plot);
  el("line", { x1: 0, x2: 0, y1: 0, y2: innerH, class: "axis" }, plot);

  if (spec.yLabel) {
    el("text", {
      transform: `translate(${14},${margin.top + innerH / 2}) rotate(-90)`,
      "text-anchor": "middle", class: "axis-label",
    }, svg).textContent = spec.yLabel;
  }
  if (spec.xLabel) {
    el("text", {
      x: margin.left + innerW / 2, y: height - 6,
      "text-anchor": "middle", class: "axis-label",
    }, svg).textContent = spec.xLabel;
  }

  const path = (points) => points
    .filter((p) => p.y !== null && isFinite(p.y) && (!isLog || p.y > 0))
    .map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`)
    .join(" ");

  /* reference markers first, so data draws over them */
  for (const marker of spec.markers || []) {
    const py = y(marker.y);
    el("line", {
      x1: 0, x2: innerW, y1: py, y2: py,
      stroke: marker.colour || faint(), "stroke-width": 1.5,
      "stroke-dasharray": "6 4", opacity: 0.9,
    }, plot);
    if (marker.label) {
      el("text", {
        x: innerW - 4, y: py - 6, "text-anchor": "end",
        class: "marker-label", fill: marker.colour || faint(),
      }, plot).textContent = marker.label;
    }
  }

  /* fitted curves */
  (spec.bands || []).forEach((band, index) => {
    const d = path(band.points);
    if (!d) return;
    el("path", {
      d, fill: "none", stroke: band.colour || palette()[index % palette().length],
      "stroke-width": 1.6, "stroke-dasharray": "5 4", opacity: 0.85,
    }, plot);
  });

  /* series */
  (spec.series || []).forEach((series, index) => {
    const colours = palette();
    const colour = series.colour || colours[index % colours.length];
    const d = path(series.points);
    if (d) {
      el("path", {
        d, fill: "none", stroke: colour,
        "stroke-width": series.width || 2.2,
        "stroke-dasharray": series.dashed ? "5 4" : null,
        "stroke-linejoin": "round", "stroke-linecap": "round",
      }, plot);
    }
    if (series.showPoints !== false) {
      for (const point of series.points) {
        if (point.y === null || !isFinite(point.y) || (isLog && point.y <= 0)) continue;

        // Error bar: a whisker spanning +/- point.err, drawn under the marker.
        // This is how the chart admits a number came from repeated trials
        // rather than a single run -- the spread is on the page, not just in
        // a tooltip.
        if (point.err && isFinite(point.err) && point.err > 0) {
          const hi = y(isLog ? Math.max(point.y + point.err, 1e-12) : point.y + point.err);
          const lo = y(isLog ? Math.max(point.y - point.err, 1e-12) : point.y - point.err);
          const cx = x(point.x);
          const cap = 4;
          el("line", { x1: cx, x2: cx, y1: hi, y2: lo, stroke: colour,
                       "stroke-width": 1.4, opacity: 0.65 }, plot);
          el("line", { x1: cx - cap, x2: cx + cap, y1: hi, y2: hi, stroke: colour,
                       "stroke-width": 1.4, opacity: 0.65 }, plot);
          el("line", { x1: cx - cap, x2: cx + cap, y1: lo, y2: lo, stroke: colour,
                       "stroke-width": 1.4, opacity: 0.65 }, plot);
        }

        const marker = el("circle", {
          cx: x(point.x), cy: y(point.y), r: series.pointRadius || 4,
          fill: colour, stroke: cssVar("--surface", "#ffffff"), "stroke-width": 1.5,
        }, plot);
        const spread = point.err ? `\n± ${formatNumber(point.err, 4)} (s.d.)` : "";
        el("title", {}, marker).textContent =
          `${series.label}\n${spec.xLabel || "x"} = ${formatNumber(point.x)}\n` +
          `${spec.yLabel || "y"} = ${formatNumber(point.y, 4)}${spread}`;
      }
    }
  });

  return svg;
}

/* ------------------------------------------------------------------ bars */
export function drawBars(container, spec) {
  container.innerHTML = "";
  const width = container.clientWidth || 640;
  const height = spec.height || 240;
  const margin = { top: 12, right: 16, bottom: 56, left: 62 };
  const innerW = Math.max(width - margin.left - margin.right, 40);
  const innerH = Math.max(height - margin.top - margin.bottom, 40);

  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height }, container);
  const plot = el("g", { transform: `translate(${margin.left},${margin.top})` }, svg);

  const values = spec.bars.map((b) => b.value).filter((v) => isFinite(v));
  if (!values.length) return;
  const lo = Math.min(0, ...values);
  const hi = Math.max(...values, 0);
  const y = linearScale(spec.yDomain || [lo, hi * 1.1 || 1], [innerH, 0]);

  for (const tick of y.ticks(5)) {
    const py = y(tick);
    el("line", { x1: 0, x2: innerW, y1: py, y2: py, class: "grid" }, plot);
    el("text", { x: -10, y: py + 4, "text-anchor": "end", class: "tick" }, plot)
      .textContent = spec.yTickFormat ? spec.yTickFormat(tick) : formatNumber(tick);
  }

  const slot = innerW / spec.bars.length;
  const barWidth = Math.min(slot * 0.62, 74);
  spec.bars.forEach((bar, index) => {
    const centre = slot * (index + 0.5);
    const top = y(Math.max(bar.value, 0));
    const bottom = y(Math.min(bar.value, 0));
    const themed = palette();
    const roleColour = bar.role === 'baseline' ? themed[1] : themed[0];
    const colour = bar.colour || (bar.role ? roleColour : themed[index % themed.length]);
    const rect = el("rect", {
      x: centre - barWidth / 2, y: top, width: barWidth,
      height: Math.max(Math.abs(bottom - top), 1), fill: colour, rx: 3, opacity: 0.92,
    }, plot);
    el("title", {}, rect).textContent = `${bar.label}: ${formatNumber(bar.value, 4)}`;
    el("text", {
      x: centre, y: top - 6, "text-anchor": "middle", class: "bar-value",
    }, plot).textContent = spec.valueFormat ? spec.valueFormat(bar.value) : formatNumber(bar.value);

    const label = el("text", {
      x: centre, y: innerH + 16, "text-anchor": "middle", class: "tick",
    }, plot);
    for (const [line, text] of String(bar.label).split("\n").entries()) {
      el("tspan", { x: centre, dy: line === 0 ? 0 : 13 }, label).textContent = text;
    }
  });

  el("line", { x1: 0, x2: innerW, y1: y(0), y2: y(0), class: "axis" }, plot);
  if (spec.yLabel) {
    el("text", {
      transform: `translate(14,${margin.top + innerH / 2}) rotate(-90)`,
      "text-anchor": "middle", class: "axis-label",
    }, svg).textContent = spec.yLabel;
  }
  return svg;
}

/* ---------------------------------------------------------------- legend */
export function renderLegend(container, entries) {
  container.innerHTML = "";
  entries.forEach((entry, index) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = `legend-swatch${entry.dashed ? " legend-swatch-dashed" : ""}`;
    const colours = palette();
    swatch.style.background = entry.colour || colours[index % colours.length];
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(entry.label));
    if (entry.note) {
      const note = document.createElement("em");
      note.textContent = ` ${entry.note}`;
      item.appendChild(note);
    }
    container.appendChild(item);
  });
}
