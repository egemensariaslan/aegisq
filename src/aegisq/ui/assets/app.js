/* AegisQ dashboard controller.
 *
 * Each panel is a small declarative unit: controls -> query string -> API call
 * -> render. Panels run independently so a slow one never blocks the page.
 */

import {
  drawChart, drawBars, renderLegend, formatNumber, formatPercent,
  PALETTE, palette, ink, faint,
} from "/assets/charts.js";

/* ------------------------------------------------------- design directions */
const STYLES = [
  { id: "instrument", label: "Instrument", hint: "dark minimalist" },
  { id: "paper", label: "Paper", hint: "editorial" },
  { id: "bento", label: "Bento", hint: "modular grid" },
  { id: "terminal", label: "Terminal", hint: "monospace" },
];
const STYLE_KEY = "aegisq.style";

/* Panels register how to redraw themselves so a style change repaints the
 * charts too. Without this the chrome would restyle and every SVG would keep
 * the previous theme's colours — the charts are drawn, not styled. */
const redraws = new Map();
const registerRedraw = (key, fn) => redraws.set(key, fn);

const DARK_STYLES = new Set(["instrument", "terminal"]);

function applyStyle(id) {
  document.documentElement.setAttribute("data-style", id);
  // Belt and suspenders: Chromium gives a :root color-scheme rule priority
  // over an equal-specificity attribute-selector rule regardless of source
  // order, so the stylesheet declarations alone are not reliable here. An
  // inline style is unambiguous and wins outright.
  document.documentElement.style.colorScheme = DARK_STYLES.has(id) ? "dark" : "light";
  try { localStorage.setItem(STYLE_KEY, id); } catch { /* private mode */ }
  for (const chip of document.querySelectorAll(".style-chip")) {
    chip.setAttribute("aria-pressed", String(chip.dataset.style === id));
  }
  // Let the new custom properties settle before re-reading them.
  requestAnimationFrame(() => {
    for (const fn of redraws.values()) {
      try { fn(); } catch { /* a panel that never ran has nothing to repaint */ }
    }
  });
}

function buildStyleSwitcher() {
  // A single segmented control (one shared track, one active pill) rather
  // than four separate coloured-dot buttons -- the standard "mode switch"
  // pattern, so the active choice reads at a glance instead of by colour.
  const host = document.getElementById("style-switcher");
  host.innerHTML = "";
  for (const style of STYLES) {
    const chip = document.createElement("button");
    chip.className = "style-chip";
    chip.dataset.style = style.id;
    chip.title = style.hint;
    chip.setAttribute("aria-pressed", "false");
    chip.textContent = style.label;
    chip.addEventListener("click", () => applyStyle(style.id));
    host.appendChild(chip);
  }
  // ?style=bento deep-links a specific look (e.g. for a README screenshot or
  // a shared link) and wins over both the saved choice and the OS preference.
  const requested = new URLSearchParams(location.search).get("style");
  let saved = null;
  try { saved = localStorage.getItem(STYLE_KEY); } catch { /* ignore */ }
  const preferred = window.matchMedia("(prefers-color-scheme: light)").matches
    ? "paper" : "instrument";
  const initial = [requested, saved].find((id) => STYLES.some((s) => s.id === id)) || preferred;
  applyStyle(initial);
}

/* ------------------------------------------------------------- utilities */
const $ = (id) => document.getElementById(id);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function fetchJSON(path, params) {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(query ? `${path}?${query}` : path);
  const payload = await response.json();
  if (!response.ok || payload.error) throw new Error(payload.error || response.statusText);
  return payload;
}

/* Wires a card-header "Copy" button. Returns a setter the panel calls after
 * every run with the command that reproduces what's on screen -- reproducing
 * a result is then one click (copies to the clipboard) rather than
 * triple-click-selecting text out of a code block under the chart. */
function copyButton(id) {
  const button = $(id);
  const label = button.querySelector(".label");
  let payload = "";
  let resetTimer = null;

  // Legacy synchronous fallback for when the async Clipboard API is
  // unavailable (a non-secure --host, or a browser that gates it behind a
  // permission this click didn't carry). No dialogs: a prompt() here would
  // block the whole page until dismissed, which is a worse failure than a
  // copy that silently didn't happen -- the command is still readable from
  // this button's title tooltip either way.
  function legacyCopy(text) {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.focus();
    area.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    area.remove();
    return ok;
  }

  button.addEventListener("click", async () => {
    if (!payload) return;
    let copied = false;
    try {
      await navigator.clipboard.writeText(payload);
      copied = true;
    } catch {
      copied = legacyCopy(payload);
    }
    if (!copied) return; // command is still on the title tooltip
    button.classList.add("copied");
    label.textContent = "Copied";
    clearTimeout(resetTimer);
    resetTimer = setTimeout(() => {
      button.classList.remove("copied");
      label.textContent = "Copy";
    }, 1600);
  });

  return (command, params) => {
    const paramText = params
      ? Object.entries(params)
          .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(",") : v}`)
          .join(" ")
      : "";
    payload = paramText ? `${command}  # ${paramText}` : command;
    button.disabled = false;
    button.title = payload;
  };
}

function stat(container, key, value, note, tone) {
  const box = element("div", "stat");
  box.appendChild(element("div", "k", key));
  box.appendChild(element("div", `v${tone ? ` ${tone}` : ""}`, value));
  if (note) box.appendChild(element("div", "n", note));
  container.appendChild(box);
}

/* Builds a control strip and returns a function reading its current values. */
function controls(container, definitions, onChange) {
  container.innerHTML = "";
  const readers = {};

  for (const def of definitions) {
    const wrap = element("div", "control");
    wrap.appendChild(element("label", null, def.label));

    if (def.type === "range") {
      const row = element("div", "range-row");
      const input = document.createElement("input");
      Object.assign(input, { type: "range", min: def.min, max: def.max, step: def.step,
                             value: def.value });
      const readout = element("span", "range-value", def.format(def.value));
      input.addEventListener("input", () => { readout.textContent = def.format(+input.value); });
      input.addEventListener("change", onChange);
      row.appendChild(input);
      row.appendChild(readout);
      wrap.appendChild(row);
      readers[def.key] = () => +input.value;
    } else if (def.type === "select") {
      const select = document.createElement("select");
      for (const option of def.options) {
        const node = element("option", null, option.label);
        node.value = option.value;
        if (String(option.value) === String(def.value)) node.selected = true;
        select.appendChild(node);
      }
      select.addEventListener("change", onChange);
      wrap.appendChild(select);
      readers[def.key] = () => select.value;
    } else if (def.type === "chips") {
      // A comma-separated numeric list (register widths, noise strengths) as
      // removable tags plus a small "add" field, rather than a raw CSV string
      // in a text box -- the underlying value is still a comma-joined string,
      // so the API and its query-string parsing are untouched.
      let values = def.value.split(",").map((v) => v.trim()).filter(Boolean);
      const list = element("div", "chip-list");
      const addInput = document.createElement("input");
      Object.assign(addInput, { type: "text", className: "chip-add",
                                placeholder: "+ add", inputMode: "decimal" });
      addInput.setAttribute("aria-label", `add a value to ${def.label}`);

      const renderChips = () => {
        values = [...new Set(values)].sort((a, b) => parseFloat(a) - parseFloat(b));
        list.querySelectorAll(".chip-tag").forEach((node) => node.remove());
        for (const value of values) {
          const tag = element("span", "chip-tag", value);
          const remove = document.createElement("button");
          remove.type = "button";
          remove.textContent = "×";
          remove.setAttribute("aria-label", `remove ${value}`);
          remove.disabled = values.length <= 1;
          remove.addEventListener("click", () => {
            if (values.length <= 1) return; // at least one point, always
            values = values.filter((v) => v !== value);
            renderChips();
            onChange();
          });
          tag.appendChild(remove);
          list.insertBefore(tag, addInput);
        }
      };
      addInput.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        const raw = addInput.value.trim();
        if (raw && !Number.isNaN(parseFloat(raw))) {
          values.push(raw);
          addInput.value = "";
          renderChips();
          onChange();
        }
      });
      list.appendChild(addInput);
      renderChips();
      wrap.appendChild(list);
      readers[def.key] = () => values.join(",");
    } else {
      const input = document.createElement("input");
      Object.assign(input, { type: def.type || "number", value: def.value });
      if (def.min !== undefined) input.min = def.min;
      if (def.max !== undefined) input.max = def.max;
      if (def.step !== undefined) input.step = def.step;
      input.addEventListener("change", onChange);
      wrap.appendChild(input);
      readers[def.key] = () => (def.type === "text" ? input.value : +input.value);
    }
    container.appendChild(wrap);
  }

  const action = element("div", "control");
  action.appendChild(element("label", null, " "));
  const button = element("button", null, "Run");
  action.appendChild(button);
  container.appendChild(action);

  const statusWrap = element("div", "control");
  statusWrap.appendChild(element("label", null, " "));
  const status = element("div", "status");
  // Screen-reader users get "running..." and the completion time announced
  // without moving focus back here after every Run.
  status.setAttribute("aria-live", "polite");
  status.setAttribute("role", "status");
  statusWrap.appendChild(status);
  container.appendChild(statusWrap);

  return {
    values: () => Object.fromEntries(Object.entries(readers).map(([k, r]) => [k, r()])),
    button,
    setBusy(busy, message) {
      button.disabled = busy;
      status.className = "status";
      status.innerHTML = "";
      if (busy) {
        status.appendChild(element("span", "spinner"));
        status.appendChild(document.createTextNode(message || "running…"));
      } else if (message) {
        status.textContent = message;
      }
    },
    setError(message) {
      button.disabled = false;
      status.className = "status error";
      status.textContent = message;
    },
  };
}

/* Wires a panel: build controls, run on demand, render or report failure. */
function panel({ node, definitions, run }) {
  const ui = controls(node, definitions, () => execute());
  async function execute() {
    const started = performance.now();
    ui.setBusy(true);
    try {
      await run(ui.values(), ui);
      ui.setBusy(false, `${((performance.now() - started) / 1000).toFixed(1)}s`);
    } catch (error) {
      ui.setError(String(error.message || error));
    }
  }
  ui.button.addEventListener("click", execute);
  return execute;
}

/* --------------------------------------------------------------- 0. env */
async function loadEnvironment() {
  const env = $("env");
  env.innerHTML = "";
  try {
    const data = await fetchJSON("/api/env");
    const shown = ["aegisq", "pennylane", "torch", "python", "platform"];
    for (const key of shown) {
      if (!data[key]) continue;
      const item = element("span", "env-badge");
      item.appendChild(element("b", null, key));
      item.appendChild(document.createTextNode(data[key]));
      env.appendChild(item);
    }
  } catch (error) {
    env.textContent = `could not read environment: ${error.message}`;
  }
}

/* ---------------------------------------------------------------- 1. ZNE */
const updateZneCommand = copyButton("zne-copy");
const zneRun = panel({
  node: $("zne-controls"),
  definitions: [
    { key: "noise", label: "depolarizing p", type: "range", min: 0.001, max: 0.05,
      step: 0.001, value: 0.01, format: (v) => v.toFixed(3) },
    { key: "layers", label: "layers", type: "number", value: 3, min: 1, max: 6 },
    { key: "qubits", label: "qubits", type: "number", value: 4, min: 2, max: 6 },
    { key: "scales", label: "scale factors", type: "select", value: "1,2,3", options: [
      { value: "1,2", label: "1, 2" },
      { value: "1,2,3", label: "1, 2, 3" },
      { value: "1,3,5", label: "1, 3, 5" },
      { value: "1,2,3,4", label: "1, 2, 3, 4" },
      { value: "1,1.5,2,2.5,3", label: "1, 1.5, 2, 2.5, 3 (unstable)" },
    ] },
    { key: "trials", label: "trials", type: "number", value: 8, min: 1, max: 30 },
  ],
  async run(values) {
    const data = await fetchJSON("/api/zne", values);
    const truth = data.truth;

    const stats = $("zne-stats");
    stats.innerHTML = "";
    stat(stats, "noiseless", formatNumber(truth, 4),
         `± ${formatNumber(data.truth_std, 3)} · n=${data.trials}`);
    stat(stats, "unmitigated", formatNumber(data.points[0].y, 4),
         `err ${formatNumber(data.raw_error, 3)} ± ${formatNumber(data.raw_error_std, 3)}`, "bad");
    for (const fit of data.fits) {
      const tone = fit.bias_reduction > 0.5 ? "good" : fit.bias_reduction > 0 ? "warn" : "bad";
      const coverage = fit.coverage < 1 ? ` · ${(fit.coverage * 100).toFixed(0)}% valid` : "";
      stat(stats, fit.label, `${formatNumber(fit.estimate, 4)} ± ${formatNumber(fit.estimate_std, 3)}`,
           `${formatPercent(fit.bias_reduction)} ± ${(fit.bias_reduction_std * 100).toFixed(0)}pp` +
           (fit.variance_cost ? ` · ${fit.variance_cost.toFixed(1)}× var` : "") + coverage, tone);
    }

    const render = () => {
    const colours = { richardson: PALETTE[0], linear: PALETTE[4], exponential: PALETTE[2] };
    drawChart($("zne-chart"), {
      height: 330,
      xLabel: "noise scale factor λ  (0 = the zero-noise limit)",
      yLabel: `${data.observable_label}`,
      xDomain: [0, Math.max(...data.params.scale_factors) * 1.05],
      xTicks: [0, ...data.params.scale_factors],
      markers: [{ y: truth, label: `noiseless  ${formatNumber(truth, 4)}`, colour: faint() }],
      bands: data.fits.filter((f) => f.curve).map((f) => ({
        label: f.label, points: f.curve, colour: colours[f.key],
      })),
      series: [
        { label: "measured", points: data.points, colour: ink(), width: 2.6, pointRadius: 5.5 },
        ...data.fits.map((f) => ({
          label: `${f.label} estimate`,
          points: [{ x: 0, y: f.estimate, err: f.estimate_std }],
          colour: colours[f.key], pointRadius: 6, showPoints: true,
        })),
      ],
    });

    renderLegend($("zne-legend"), [
      { label: "measured (mean ± s.d.)", colour: ink() },
      ...data.fits.map((f) => ({
        label: f.label, colour: colours[f.key],
        note: `→ ${formatNumber(f.estimate, 4)} ± ${formatNumber(f.estimate_std, 3)}`,
      })),
      { label: "noiseless", colour: faint(), dashed: true },
    ]);

    updateZneCommand(data.command, data.params);
    };
    render();
    registerRedraw("zne", render);
  },
});

/* ------------------------------------------------------------ 2. plateau */
const updatePlateauCommand = copyButton("plateau-copy");
const plateauRun = panel({
  node: $("plateau-controls"),
  definitions: [
    { key: "qubits", label: "widths", type: "chips", value: "4,6,8,10" },
    { key: "layers", label: "layers", type: "number", value: 4, min: 1, max: 8 },
    { key: "samples", label: "samples", type: "number", value: 25, min: 5, max: 120, step: 5 },
  ],
  async run(values) {
    const data = await fetchJSON("/api/plateau", values);

    const render = () => {
    drawChart($("plateau-chart"), {
      height: 320,
      yScale: "log",
      xLabel: "qubits",
      yLabel: "mean gradient variance",
      xTicks: data.params.qubit_counts,
      xTickFormat: (v) => String(Math.round(v)),
      yTickFormat: (v) => v.toExponential(0),
      series: data.series.map((s) => ({ label: s.label, role: s.role, points: s.points })),
    });

    renderLegend($("plateau-legend"), data.series.map((s) => ({
      label: s.label, role: s.role,
      note: `×${formatNumber(s.per_qubit_factor, 3)} per qubit`,
    })));

    const table = $("plateau-table");
    table.innerHTML = "";
    const head = table.createTHead().insertRow();
    ["ansatz", ...data.params.qubit_counts.map((n) => `n=${n}`),
     "decay", "per qubit", "dead angles"].forEach((title) => {
      head.appendChild(element("th", null, title));
    });
    const body = table.createTBody();
    for (const series of data.series) {
      const row = body.insertRow();
      const name = row.insertCell();
      name.textContent = series.label;
      // Only the external references are marked; untagged rows are this library's.
      if (series.role === "baseline") {
        name.appendChild(element("span", "tag", "reference"));
      }
      for (const point of series.points) {
        row.insertCell().textContent = point.y === null ? "—" : point.y.toExponential(2);
      }
      row.insertCell().textContent = formatNumber(series.decay, 3);
      row.insertCell().textContent = `×${formatNumber(series.per_qubit_factor, 3)}`;
      row.insertCell().textContent =
        `${(series.points[series.points.length - 1].dead * 100).toFixed(0)}%`;
    }
    updatePlateauCommand(data.command, data.params);
    };
    render();
    registerRedraw("plateau", render);
  },
});

/* -------------------------------------------------------- 3. noise sweep */
const updateSweepCommand = copyButton("sweep-copy");
const sweepRun = panel({
  node: $("sweep-controls"),
  definitions: [
    { key: "qubits", label: "qubits", type: "number", value: 4, min: 2, max: 6 },
    { key: "layers", label: "layers", type: "number", value: 3, min: 1, max: 6 },
    { key: "strengths", label: "depolarizing p", type: "chips",
      value: "0,0.0025,0.005,0.01,0.02,0.04" },
    { key: "trials", label: "trials", type: "number", value: 5, min: 1, max: 20 },
  ],
  async run(values) {
    const data = await fetchJSON("/api/noise-sweep", values);
    const render = () => {
    drawChart($("sweep-chart"), {
      height: 320,
      xLabel: "depolarizing probability per gate",
      yLabel: "signal retained  f  (mean ± s.d.)",
      xTickFormat: (v) => (v === 0 ? "0" : v.toFixed(3)),
      markers: [{ y: 1, label: "noiseless", colour: faint() }],
      series: data.series.map((s) => ({
        label: s.label, role: s.role, points: s.points, dashed: s.mitigated,
      })),
    });
    renderLegend($("sweep-legend"), data.series.map((s) => {
      const last = s.points[s.points.length - 1];
      return {
        label: s.label, role: s.role, dashed: s.mitigated,
        note: `f ${formatNumber(last.y, 2)} ± ${formatNumber(last.err, 2)}  (n=${data.trials})`,
      };
    }));
    updateSweepCommand(data.command, data.params);
    };
    render();
    registerRedraw("sweep", render);
  },
});

/* ----------------------------------------------------------- 4. symmetry */
const updateSymmetryCommand = copyButton("symmetry-copy");
const symmetryRun = panel({
  node: $("symmetry-controls"),
  definitions: [
    { key: "qubits", label: "qubits", type: "number", value: 4, min: 3, max: 6 },
    { key: "layers", label: "layers", type: "number", value: 3, min: 1, max: 6 },
    { key: "strengths", label: "depolarizing p", type: "chips", value: "0.002,0.005,0.01,0.02,0.05" },
  ],
  async run(values) {
    const data = await fetchJSON("/api/symmetry", values);
    const render = () => {
    const stats = $("symmetry-stats");
    stats.innerHTML = "";
    stat(stats, "conserved", data.symmetry || "—",
         `${data.params.n_particles} of ${data.params.n_qubits} qubits`);
    stat(stats, "sector", `${data.sector_size} / ${data.space_size}`, "basis states");
    stat(stats, "leakage at p = 0", data.noiseless_leakage.toExponential(1), "");

    drawChart($("symmetry-chart"), {
      height: 260,
      xLabel: "depolarizing probability per gate",
      yLabel: "probability mass",
      yDomain: [0, 1],
      xTickFormat: (v) => v.toFixed(3),
      series: [
        { label: "leaked out of sector",
          points: data.rows.map((r) => ({ x: r.x, y: r.leakage })), colour: PALETTE[1] },
        { label: "accepted (in sector)",
          points: data.rows.map((r) => ({ x: r.x, y: r.accepted })), colour: PALETTE[2] },
      ],
    });
    renderLegend($("symmetry-legend"), [
      { label: "leaked", colour: PALETTE[1] },
      { label: "accepted", colour: PALETTE[2] },
    ]);

    drawBars($("symmetry-bars"), {
      height: 260,
      yLabel: "mean absolute error",
      valueFormat: (v) => formatNumber(v, 2),
      bars: data.rows.flatMap((row) => ([
        { label: `${row.x}\nraw`, value: row.raw_error, colour: PALETTE[1] },
        { label: `\nver.`, value: row.verified_error, colour: PALETTE[2] },
      ])),
    });
    updateSymmetryCommand(data.command, data.params);
    };
    render();
    registerRedraw("symmetry", render);
  },
});

/* ----------------------------------------------------------- 5. training */
let trainingSource = null;

const updateTrainCommand = copyButton("train-copy");
const trainRun = panel({
  node: $("train-controls"),
  definitions: [
    { key: "epochs", label: "epochs", type: "number", value: 10, min: 1, max: 40 },
    { key: "qubits", label: "qubits", type: "number", value: 4, min: 2, max: 6 },
    { key: "noise", label: "noise model", type: "select", value: "hardware_like", options: [
      { value: "hardware_like", label: "hardware_like" },
      { value: "depolarizing", label: "depolarizing" },
      { value: "thermal_relaxation", label: "thermal" },
      { value: "noiseless", label: "noiseless" },
    ] },
    { key: "dataset", label: "dataset", type: "select", value: "two_moons", options: [
      { value: "two_moons", label: "two_moons" },
      { value: "circles", label: "circles" },
      { value: "parity", label: "parity" },
    ] },
  ],
  run(values, ui) {
    return new Promise((resolve, reject) => {
      $("train-chart").innerHTML = '<p class="placeholder">training…</p>';
      if (trainingSource) trainingSource.close();
      const query = new URLSearchParams(values).toString();
      const source = new EventSource(`/api/train?${query}`);
      trainingSource = source;

      const history = new Map();
      let meta = null;

      const render = () => {
        if (!meta) return;
        const series = meta.series.map((s) => ({
          label: s.label, role: s.role,
          points: (history.get(s.key) || []).map((r) => ({ x: r.epoch, y: r.test_accuracy })),
        }));
        drawChart($("train-chart"), {
          height: 300,
          xLabel: "epoch",
          yLabel: "test accuracy",
          yDomain: [0, 1.02],
          xDomain: [0, meta.epochs],
          xTickFormat: (v) => String(Math.round(v)),
          markers: [{ y: 0.5, label: "chance", colour: faint() }],
          series,
        });
        renderLegend($("train-legend"), meta.series.map((s) => {
          const rows = history.get(s.key) || [];
          const last = rows[rows.length - 1];
          return {
            label: s.label, role: s.role,
            note: last ? `test ${last.test_accuracy.toFixed(3)} · ${s.two_qubit_gates} 2q gates` : "",
          };
        }));

        const stats = $("train-stats");
        stats.innerHTML = "";
        stat(stats, "noise model", meta.noise.split(" ")[0], meta.noise);
        stat(stats, "dataset", meta.dataset.name,
             `${meta.dataset.train} train / ${meta.dataset.test} test`);
        for (const s of meta.series) {
          const rows = history.get(s.key) || [];
          const last = rows[rows.length - 1];
          if (!last) continue;
          stat(stats, s.label, last.test_accuracy.toFixed(3),
               `epoch ${last.epoch} · train ${last.train_accuracy.toFixed(3)}`,
               last.test_accuracy > 0.85 ? "good" : last.test_accuracy > 0.65 ? "warn" : "bad");
        }
      };

      source.onmessage = (message) => {
        const event = JSON.parse(message.data);
        if (event.event === "start") {
          meta = event;
          history.clear();
          updateTrainCommand("python3 run.py benchmark --quick", values);
          registerRedraw("train", render);
          render();
        } else if (event.event === "epoch") {
          if (!history.has(event.key)) history.set(event.key, []);
          history.get(event.key).push(event);
          ui.setBusy(true, `epoch ${event.epoch}/${meta ? meta.epochs : "?"}`);
          render();
        } else if (event.event === "done") {
          source.close();
          trainingSource = null;
          resolve();
        } else if (event.event === "error") {
          source.close();
          trainingSource = null;
          reject(new Error(event.message));
        }
      };
      source.onerror = () => {
        source.close();
        trainingSource = null;
        // A completed stream also fires onerror once the server closes it.
        if (meta) resolve(); else reject(new Error("training stream failed"));
      };
    });
  },
});

/* ------------------------------------------------------------ 6. catalog */
const catalogRun = panel({
  node: $("catalog-controls"),
  definitions: [
    { key: "qubits", label: "qubits", type: "number", value: 6, min: 2, max: 12 },
    { key: "layers", label: "layers", type: "number", value: 1, min: 1, max: 6 },
  ],
  async run(values) {
    const data = await fetchJSON("/api/catalog", values);
    const render = () => {
    const table = $("catalog-table");
    table.innerHTML = "";
    const head = table.createTHead().insertRow();
    ["ansatz", "2q gates", "parameters", "2q depth", "conserves"].forEach((title) => {
      head.appendChild(element("th", null, title));
    });
    const body = table.createTBody();
    for (const row of data.rows) {
      const tr = body.insertRow();
      const name = tr.insertCell();
      name.textContent = row.label;
      if (row.role === "baseline") {
        name.appendChild(element("span", "tag", "reference"));
      }
      tr.insertCell().textContent = row.two_qubit_gates;
      tr.insertCell().textContent = row.parameters;
      tr.insertCell().textContent = row.depth;
      tr.insertCell().textContent = row.symmetry || "—";
    }
    drawBars($("catalog-chart"), {
      height: 230,
      yLabel: "two-qubit depth per layer",
      valueFormat: (v) => String(v),
      bars: data.rows.map((row) => ({
        label: row.label.replace(/([a-z])([A-Z])/g, "$1\n$2"),
        value: row.depth, role: row.role,
      })),
    });
    };
    render();
    registerRedraw("catalog", render);
  },
});

/* ------------------------------------------------------------------ boot */
async function boot() {
  buildStyleSwitcher();
  $("train-chart").innerHTML = '<p class="placeholder">Run to train.</p>';
  await loadEnvironment();
  // Cheap panels first so the page has content while the slower ones fill in.
  await catalogRun();
  await zneRun();
  await symmetryRun();
  await sweepRun();
  await plateauRun();
}

boot();
